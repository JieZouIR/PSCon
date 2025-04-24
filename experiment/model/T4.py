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

