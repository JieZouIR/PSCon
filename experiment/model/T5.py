import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Utils import *
import os


class T5(nn.Module):
    """
    T5 model implementation for passage selection task
    This model uses T5 encoder to process context and passage, then predicts the relevance score
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
        Process T5 output to get selected passages
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

    def forward(self, context, passage, common_output):
        """
        Forward pass of the model
        Args:
            context: Input context
            passage: Input passage
            common_output: Common output from previous layers
        Returns:
            Updated common_output with T5 predictions
        """
        batch_size, context_len = context.size()
        # Get context embeddings
        context_emb = self.embedding(context)
        features = []
        # Process passage if exists
        if passage is not None and "passage_context_states" not in common_output:
            batch_size, num_passages, passage_len = passage.size()
            passage_emb = self.embedding(passage)

            # Concatenate passage and context
            passage_context = torch.cat([passage, context.unsqueeze(1).expand(-1, num_passages, -1)], dim=-1).reshape(
                batch_size * num_passages, passage_len + context_len)
            passage_context_emb = torch.cat([passage_emb, context_emb.unsqueeze(1).expand(-1, num_passages, -1, -1)],
                                            dim=-2).reshape(batch_size * num_passages, passage_len + context_len, -1)
            
            # Process through T5 encoder
            passage_context_states, passage_context_weights = self.t5_encoder(passage_context_emb,
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
        # passage_context_states 4 * [batch_size, num_queries, passage_len + context_len, hidden_size]
        # paramater [batch_size, num_passages, hidden_size], 0 means P-CLS of passage
        # t5_output [batch_size, num_passages]
        t5_output = self.linear(common_output['passage_context_states'][-1][:, :, 0]).squeeze(-1)
        t5_output = self.softmax(t5_output)

        common_output['t5_output'] = t5_output

        return common_output

    def loss(self, t5_output, selected_passage, passage_loss_mask):
        """
        Calculate loss for T5 model
        Args:
            t5_output: Model predictions
            selected_passage: Ground truth selected passages
            passage_loss_mask: Mask for valid passages
        Returns:
            Computed loss value
        """
        # passage_loss_mask [batch_size, 1]
        # t5_output&selected_passage [batch_size, num_passages]
        t5_loss = (passage_loss_mask.detach() * F.cross_entropy(t5_output, selected_passage.float(), reduction='none').mean(dim=-1, keepdim=True) + 1e-8).sum() / (passage_loss_mask.detach().sum() + 1)
        
        return t5_loss



