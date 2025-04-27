import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class T1(nn.Module):
    """
    T1 model for intent detection
    This model takes context as input and predicts the intent
    """
    def __init__(self, embedding, t1_encoder, hidden_size, id2intent):
        super(T1, self).__init__()

        # Initialize model parameters
        self.hidden_size=hidden_size
        self.id2intent=id2intent

        # Initialize model components
        self.embedding=embedding
        self.t1_encoder=t1_encoder

        # Linear layer for intent detection
        self.linear = nn.Linear(hidden_size, len(id2intent), bias=False)

    def label(self, t1_output):
        # t1_output [batch_size], translate intent_id to the corresponding intent
        # format of returned data is ["intent1", "intent2", ...]
        return [self.id2intent[t1_output[i].item()] for i in range(t1_output.size(0))]

    def forward(self, context, common_output):
        """
        Forward pass of the T1 model
        Args:
            context: input context sequence
            common_output: dictionary to store intermediate outputs
        Returns:
            common_output: updated with context states, weights and T1 output
        """
        # context_emb [batch_size, context_len, hidden_size]
        context_emb = self.embedding(context)

        # context_states 4 * [batch_size, context_len, hidden_size]
        # context_weights 4 * [batch_size, context_len, context_len]
        context_states, context_weights = self.t1_encoder(context_emb, src_key_padding_mask=context.eq(0))
        
        # # Save intermediate outputs for debugging
        # current_directory = os.getcwd()
        # output_file_path = os.path.join(current_directory, "T1out.txt")
        # with open(output_file_path, "a") as file:
        #     file.write(f"context_states:{context_states}\n")
        #     file.write(f"context_states[-1][:, 0]:{context_states[-1][:, 0]}\n\n\n")
        
        # -1 means the last layer of transformers
        # parameters [batch_size, hidden_size], 0 means I-CLS of context
        # t1_output [batch_size, num_intent]
        t1_output = self.linear(context_states[-1][:, 0])

        # Store outputs in common_output dictionary
        common_output['context_states'] = context_states
        common_output['context_weights'] = context_weights
        common_output['t1_output']=t1_output
        return common_output

    def loss(self, t1_output, intent):
        """
        Calculate cross entropy loss for intent detection
        Args:
            t1_output: model predictions
            intent: ground truth intent labels
        Returns:
            t1_loss: cross entropy loss
        """
        # print(f"shape2:{intent.squeeze(dim=1)}")
        t1_loss = F.cross_entropy(t1_output, intent.reshape(-1))
        return t1_loss


