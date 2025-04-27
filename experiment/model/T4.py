import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Utils import *
import os

class T4(nn.Module):
    """
    T4 model for query ranking
    """
    def __init__(self, embedding, t4_encoder, hidden_size):
        super(T4, self).__init__()

        self.hidden_size=hidden_size

        self.embedding=embedding
        self.t4_encoder=t4_encoder

        self.linear = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def label(self, t4_output):
        """
        Process the T4 model output to select top-k queries
        Args:
            t4_output: [batch_size, num_queries], binary sequence where 1 indicates selected query
        Returns:
            List of selected query indices for each sample in the batch
        """
        selected = []
        k = 5  # Select top-k querys

        for i in range(t4_output.size(0)):
            selected = []
            for i in range(t4_output.size(0)):
                rs = []
                for j in range(t4_output.size(1)):
                    if t4_output[i, j]:
                        rs.append(j)
                selected.append(rs)
            # return selected
        return t4_output

    def forward(self, context, query, common_output):
        """
        Forward pass of the T4 model
        Args:
            context: Input context
            query: Input query
            common_output: Dictionary containing intermediate outputs
        Returns:
            Updated common_output with T4 model predictions
        """
        batch_size, context_len = context.size()
        # Get context embeddings
        context_emb = self.embedding(context)
        features = []
        
        # Process query if exists
        if query is not None and "query_context_states" not in common_output:
            batch_size, num_queries, query_len = query.size()
            # Get query embeddings
            query_emb = self.embedding(query)
            # Concatenate query and context for processing
            query_context = torch.cat([query, context.unsqueeze(1).expand(-1, num_queries, -1)], dim=-1).reshape(
                batch_size * num_queries, query_len + context_len)
            
            # Get embeddings for concatenated query and context
            query_context_emb = torch.cat([query_emb, context_emb.unsqueeze(1).expand(-1, num_queries, -1, -1)],
                                          dim=-2).reshape(batch_size * num_queries, query_len + context_len, -1)
            
            # Process through T4 encoder
            # query_context_states: 4 * [batch_size, num_queries, query_len + context_len, hidden_size]
            # query_context_weights: 4 * [batch_size, num_queries, query_len + context_len, query_len + context_len]
            query_context_states, query_context_weights = self.t4_encoder(query_context_emb,
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
        # Process query context states and generate predictions
        t4_output = self.linear(common_output['query_context_states'][-1][:, :, 0]).squeeze(-1)
        t4_output = self.softmax(t4_output)
        common_output['t4_output'] = t4_output

        return common_output

    def loss(self, t4_output, selected_query, query_loss_mask):
        """
        Calculate the loss for the T4 model
        Args:
            t4_output: Model predictions [batch_size, num_queries]
            selected_query: Ground truth selected queries [batch_size, num_queries]
            query_loss_mask: Mask for valid queries [batch_size, 1]
        Returns:
            Computed loss value
        """
        # Calculate cross entropy loss with masking
        t4_loss = (query_loss_mask.detach() * F.cross_entropy(t4_output, selected_query.float(), reduction='none').mean(dim=-1, keepdim=True) + 1e-8).sum() / (query_loss_mask.detach().sum() + 1)

        return t4_loss

