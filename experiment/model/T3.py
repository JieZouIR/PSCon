import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class T3(nn.Module):
    """
    T3 model for action prediction
    This model takes context, query, and passage as input and predicts the action
    """
    def __init__(self, embedding, t3_encoder, hidden_size, id2action):
        super(T3, self).__init__()

        self.hidden_size = hidden_size
        self.id2action = id2action

        self.embedding = embedding
        self.t3_encoder = t3_encoder

        # Linear layer for action prediction
        self.linear = nn.Linear(hidden_size, len(id2action), bias=False)
        

    def label(self, t3_output):
        # t3_output [batch_size], translate action_id to corresponding action
        # format of return ["action1", "action2", ...]
        return [self.id2action[t3_output[i].item()] for i in range(t3_output.size(0))]

    def forward(self, context, query, passage, common_output):
        # Get batch size and context length
        batch_size, context_len = context.size()
        # Get context embeddings
        context_emb = self.embedding(context)
        # context_states 4 * [batch_size, context_len, hidden_size]
        # context_weights 4 * [batch_size, context_len, context_len]
        context_states, context_weights = self.t3_encoder(context_emb, src_key_padding_mask=context.eq(0))
        features = []
        
        # Process query if exists
        if query is not None:
            batch_size, num_queries, query_len = query.size()
            # Get query embeddings
            query_emb = self.embedding(query)
            # Concatenate query and context for processing
            query_context = torch.cat([query, context.unsqueeze(1).expand(-1, num_queries, -1)], dim=-1).reshape(
                batch_size * num_queries, query_len + context_len)
            
            # Get embeddings for concatenated query and context
            query_context_emb = torch.cat([query_emb, context_emb.unsqueeze(1).expand(-1, num_queries, -1, -1)],
                                          dim=-2).reshape(batch_size * num_queries, query_len + context_len, -1)
            
            # Process through T3 encoder
            # query_context_states: 4 * [batch_size, num_queries, query_len + context_len, hidden_size]
            # query_context_weights: 4 * [batch_size, num_queries, query_len + context_len, query_len + context_len]
            query_context_states, query_context_weights = self.t3_encoder(query_context_emb,
                                                                          src_key_padding_mask=query_context.eq(0))
            
            # Reshape states and weights for each layer
            for i in range(len(query_context_states)):
                query_context_states[i] = query_context_states[i].reshape(batch_size, num_queries,
                                                                          query_len + context_len, -1)
                query_context_weights[i] = query_context_weights[i].reshape(batch_size, num_queries,
                                                                            query_len + context_len, -1)
            
            # Extract features from the last layer's CLS token
            features.append(query_context_states[-1][:, :, query_len + 1])
            common_output['query_context_states'] = query_context_states
            common_output['query_context_weights'] = query_context_weights
            
        # Process passage if exists
        if passage is not None:
            batch_size, num_passages, passage_len = passage.size()
            passage_emb = self.embedding(passage)

            # Concatenate passage and context
            passage_context = torch.cat([passage, context.unsqueeze(1).expand(-1, num_passages, -1)], dim=-1).reshape(
                batch_size * num_passages, passage_len + context_len)
            passage_context_emb = torch.cat([passage_emb, context_emb.unsqueeze(1).expand(-1, num_passages, -1, -1)],
                                            dim=-2).reshape(batch_size * num_passages, passage_len + context_len, -1)
            
            # Process through T3 encoder
            passage_context_states, passage_context_weights = self.t3_encoder(passage_context_emb,
                                                                              src_key_padding_mask=passage_context.eq(0))
            
            # Reshape states and weights for each layer
            for i in range(len(passage_context_states)):
                passage_context_states[i] = passage_context_states[i].reshape(batch_size, num_passages,
                                                                              passage_len + context_len, -1)
                passage_context_weights[i] = passage_context_weights[i].reshape(batch_size, num_passages,
                                                                                passage_len + context_len, -1)
            
            # Extract features from the last layer's CLS token
            features.append(passage_context_states[-1][:, :, passage_len + 1])
            common_output['passage_context_states'] = passage_context_states
            common_output['passage_context_weights'] = passage_context_weights
            
        # Store outputs in common_output dictionary
        common_output['context_states'] = context_states
        common_output['context_weights'] = context_weights
        # Get final prediction by taking max over features and applying linear layer
        t3_output = self.linear(torch.cat(features, dim=1).max(dim=1, keepdim=False)[0])
        common_output['t3_output'] = t3_output
        return common_output

    def loss(self, t3_output, action):
        # Calculate cross entropy loss for action prediction
        t3_loss = F.cross_entropy(t3_output, action.reshape(-1))
        return t3_loss
