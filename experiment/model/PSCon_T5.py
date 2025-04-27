import torch.nn as nn
import torch
import copy

class PSCon_T5Model(nn.Module):
    def __init__(self, t5, response_len=50):
        super(PSCon_T5Model, self).__init__()
        self.response_len = response_len
        self.t5 = t5

    def passage(self, t5_output):
        return self.t5.label(t5_output)

    def do_forward(self, context, query, passage, response, common_output):
        if 't5_output' not in common_output:
            self.t5(context, passage, common_output)
        return common_output

    def forward(self, data, method):
        if method=='train':
            common_output = {'selected_query': data['selected_query'], 'selected_passage': data['selected_passage'],
                             'method': 'train'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            # t5_output [batch_size, num_passages]    selected_passage [batch_size, passages]    passage_loss_mask [batch_size, 1]
            t5_loss = self.t5.loss(output['t5_output'], data['selected_passage'], data['passage_loss_mask'])
            return {'t5_loss': t5_loss}
            
        elif method == 'test':
            common_output = {'method': 'test'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            return_output = {}
            # t5_output [batch_size, num_passages]
            return_output['t5_output'] = output['t5_output']
            return return_output





