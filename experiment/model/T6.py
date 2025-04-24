import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Utils import *

class T6(nn.Module):
    """
    T6 model for response generation with context, query and product information
    Uses a decoder architecture with attention mechanism
    """
    def __init__(self, embedding, t6_decoder, t6_generator, hidden_size, id2vocab):
        super(T6, self).__init__()

        self.hidden_size=hidden_size
        self.id2vocab=id2vocab

        self.embedding=embedding
        self.t6_decoder=t6_decoder

        self.t6_generator = t6_generator

    def label(self, t6_output):
        """
        Convert model output to readable text labels
        Args:
            t6_output: model output tensor [batch_size, response_len - 1]
        Returns:
            List of token sequences for each sample in batch
        """
        # t6_output [batch_size, response_len - 1]
        # format of return [['c11', 'c12', 'c13', ...],
        #                   ['c21', 'c22', 'c23', ...],
        #                   ...]
        rs = list()
        for i in range(t6_output.size(0)):
            indexes = t6_output[i]
            sent = []
            for index in indexes:
                index = index.item()
                w = self.id2vocab[index]
                if w == '[SEP]' or w == '[PAD]':
                    continue
                if w == '[EOS]':
                    break
                sent.append(w)
            if len(sent) == 0:
                sent.append('[UNK]')
            rs.append(sent)
        return rs

    def forward(self, context, query, product, response, common_output):
        """
        Forward pass of the T6 model
        Args:
            context: input context sequence [batch_size, context_len]
            query: input query sequence [batch_size, num_queries, query_len]
            product: input product sequence [batch_size, num_products, product_len]
            response: target response sequence [batch_size, response_len]
            common_output: dictionary containing shared model states
        Returns:
            Updated common_output with T6 model predictions
        """
        # context [batch_size, context_len]
        # query [batch_size, num_queries, query_len]
        # product [batch_size, num_products, product_len]
        # response [batch_size, response_len]
        batch_size, context_len = context.size()
        # context_state [batch_size, context_len, hidden_size]
        context_state = common_output['context_states'][-1]

        batch_size, response_len = response.size()
        # response_emb [batch_size, response_len, hidden_size]
        response_emb = self.embedding(response)
        # response_mask [batch_size, response_len, response_len], lower triangular values are 0, others are small numbers
        response_mask = generate_square_subsequent_mask(response_len, self.training)

        features=[]
        weights=[]
        # response_context_states 4 * [batch_size, response_len, hidden_size]
        # response_weights 4 * [batch_size, response_len, response_len]
        # context_weights 4 * [batch_size, response_len, context_len]
        response_context_states, response_weights, context_weights = self.t6_decoder(response_emb, context_state, tgt_mask=response_mask, memory_mask=None, tgt_key_padding_mask=response.eq(0), memory_key_padding_mask=context.eq(0))
        features.append(response_context_states[-1].unsqueeze(1))
        # one-hot encoding of the context
        # context_map  [batch_size, context_len, vocab_size]
        context_map=build_map(context, len(self.id2vocab))
        weights.append(torch.bmm(context_weights[-1], context_map).unsqueeze(1))

        if query is not None:
            # Process query information
            batch_size, num_queries, query_len = query.size()
            query=query.reshape(batch_size * num_queries, -1)
            # query_context_states  4 * [batch_size, num_queries, query_len + context_len, hidden_size]
            # query_state  [batch_size, num_queries, query_len, hidden_size]
            query_state = common_output['query_context_states'][-1][:, :, :query_len]
            query_state=query_state.reshape(batch_size * num_queries, query_len, -1)

            # response_query [batch_size, num_queries, response_len]
            response_query = response.unsqueeze(1).expand(-1, num_queries, -1).reshape(batch_size * num_queries, -1)
            # response_query_emb [batch_size, num_queries, response_len, hidden_size]
            response_query_emb = response_emb.unsqueeze(1).expand(-1, num_queries, -1, -1).reshape(batch_size * num_queries, response_len, -1)
            # response_query_states 4 * [batch_size * num_queries, response_len, hidden_size]
            # response_weights 4 * [batch_size * num_queries, response_len, response_len]
            # query_weights 4 * [batch_size * num_queries, response_len, query_len]
            response_query_states, response_weights, query_weights = self.t6_decoder(response_query_emb, query_state,
                                                                                     tgt_mask=response_mask,
                                                                                     memory_mask=None,
                                                                                     tgt_key_padding_mask=response_query.eq(0),
                                                                                     memory_key_padding_mask=query.eq(0))
            for i in range(len(response_query_states)):
                response_query_states[i] = response_query_states[i].reshape(batch_size, num_queries, response_len, -1)

            features.append(response_query_states[-1])
            # one-hot encoding of the query
            # query_map  [batch_size * num_queries, query_len, vocab_size]
            query_map = build_map(query, len(self.id2vocab))
            # t4_output [batch_size, num_queries]
            # q_weight = torch.bmm(common_output['t4_output'].float().reshape(-1).unsqueeze(-1).unsqueeze(-1) * query_weights[-1], query_map)
            if common_output['method'] == 'train':
                q_weight = torch.bmm(common_output['selected_query'].float().reshape(-1).unsqueeze(-1).unsqueeze(-1) * query_weights[-1],query_map)
            else:
                q_weight = torch.bmm(torch.sigmoid(common_output['t4_output'].detach()).reshape(-1).unsqueeze(-1).unsqueeze(-1) * query_weights[-1], query_map)
            # q_weight [batch_size, num_queries, response_len, vocab_size]
            q_weight = q_weight.reshape(batch_size, num_queries, response_len, -1)
            weights.append(q_weight)

        if product is not None:
            # Process product information
            batch_size, num_products, product_len = product.size()
            product = product.reshape(batch_size * num_products, -1)
            product_state = common_output['product_context_states'][-1][:, :, :product_len]
            product_state = product_state.reshape(batch_size * num_products, product_len, -1)

            response_product = response.unsqueeze(1).expand(-1, num_products, -1).reshape(batch_size * num_products, -1)
            response_product_emb = response_emb.unsqueeze(1).expand(-1, num_products, -1, -1).reshape(batch_size * num_products, response_len, -1)
            response_product_states, response_weights, product_weights = self.t6_decoder(response_product_emb, product_state,
                                                                                         tgt_mask=response_mask,
                                                                                         memory_mask=None,
                                                                                         tgt_key_padding_mask=response_product.eq(0),
                                                                                         memory_key_padding_mask=product.eq(0))
            for i in range(len(response_product_states)):
                response_product_states[i] = response_product_states[i].reshape(batch_size, num_products, response_len, -1)

            features.append(response_product_states[-1])
            # one-hot encoding of the product
            product_map = build_map(product, len(self.id2vocab))
            # p_weight = torch.bmm(common_output['t5_output'].float().reshape(-1).unsqueeze(-1).unsqueeze(-1) * product_weights[-1], product_map)
            if common_output['method'] == 'train':
                p_weight = torch.bmm(common_output['selected_product'].float().reshape(-1).unsqueeze(-1).unsqueeze(-1) * product_weights[-1], product_map)
            else:
                p_weight = torch.bmm(torch.sigmoid(common_output['t5_output'].detach()).reshape(-1).unsqueeze(-1).unsqueeze(-1) * product_weights[-1], product_map)
            # p_weight [batch_size, num_products, response_len, vocab_size]
            p_weight = p_weight.reshape(batch_size, num_products, response_len, -1)
            weights.append(p_weight)
        # t6_output_1 [batch_size, response_len, vocab_size]
        t6_output_1 = F.softmax(self.t6_generator(torch.cat(features, dim=1).max(dim=1, keepdim=False)[0]), dim=-1)
        # t6_output_2 [batch_size, response_len, vocab_size]
        t6_output_2 = torch.max(torch.cat(weights, dim=1), dim=1, keepdim=False)[0]
        # t6_output_1 is probability of generating a word, and t6_output_2 is probability of copying a word
        # t6_output [batch_size, response_len, vocab_size]
        common_output['t6_output'] = 0.5*t6_output_1 + 0.5*t6_output_2

        return common_output

    def loss(self, t6_output, response):
        """
        Calculate the loss for T6 model
        Args:
            t6_output: model predictions [batch_size, response_len - 1]
            response: target response [batch_size, response_len]
        Returns:
            Negative log likelihood loss
        """
        # t6_output [batch_size, response_len - 1]
        # response [batch_size, response_len]
        t6_loss = F.nll_loss((t6_output.reshape(-1, t6_output.size(-1))+1e-8).log(), response[:, 1:].reshape(-1), ignore_index=0)
        return t6_loss



