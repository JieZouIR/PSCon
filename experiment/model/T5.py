import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Utils import *
import os


class T5(nn.Module):
    """
    T5 model implementation for product selection task
    This model uses T5 encoder to process context and product, then predicts the relevance score
    """
    def __init__(self, embedding, t5_encoder, hidden_size):
        """
        Initialize T5 model
        Args:
            embedding: Word embedding layer
            t5_encoder: T5 encoder model
            hidden_size: Hidden dimension size
        """
        super(T5, self).__init__()

        self.hidden_size=hidden_size

        self.embedding=embedding
        self.t5_encoder=t5_encoder

        self.linear = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def label(self, t5_output):
        """
        Process T5 output to get selected products
        Args:
            t5_output: Model output tensor
        Returns:
            Processed output tensor
        """
        selected = []
        for i in range(t5_output.size(0)):
            rs = []
            for j in range(t5_output.size(1)):
                if t5_output[i, j]:
                    rs.append(j)
            selected.append(rs)
        return t5_output
        # return selected

    def forward(self, context, product, common_output):
        """
        Forward pass of the model
        Args:
            context: Input context
            product: Input product
            common_output: Common output from previous layers
        Returns:
            Updated common_output with T5 predictions
        """
        # product_context_states 4 * [batch_size, num_queries, product_len + context_len, hidden_size]
        # paramater [batch_size, num_products, hidden_size], 0 means P-CLS of product
        # t5_output [batch_size, num_products]
        t5_output = self.linear(common_output['product_context_states'][-1][:, :, 0]).squeeze(-1)
        t5_output = self.softmax(t5_output)

        common_output['t5_output'] = t5_output

        return common_output

    def loss(self, t5_output, selected_product, product_loss_mask):
        """
        Calculate loss for T5 model
        Args:
            t5_output: Model predictions
            selected_product: Ground truth selected products
            product_loss_mask: Mask for valid products
        Returns:
            Computed loss value
        """
        # product_loss_mask [batch_size, 1]
        # t5_output&selected_product [batch_size, num_products]
        t5_loss = (product_loss_mask.detach() * F.cross_entropy(t5_output, selected_product.float(), reduction='none').mean(dim=-1, keepdim=True) + 1e-8).sum() / (product_loss_mask.detach().sum() + 1)
        
        return t5_loss



