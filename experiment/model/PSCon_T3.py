import torch.nn as nn
import torch
import copy

class PSCon_T3Model(nn.Module):
    def __init__(self, t3, response_len=50):
        super(PSCon_T3Model, self).__init__()
        self.response_len = response_len
        self.t3 = t3

    def action(self, t3_output):
        return self.t3.label(t3_output)

    def do_forward(self, context, query, passage, response, common_output):
        if 't3_output' not in common_output:
            self.t3(context, query, passage, common_output)
        return common_output

    def forward(self, data, method):
        if method=='train':
            common_output = {'selected_query': data['selected_query'], 'selected_passage': data['selected_passage'],
                             'method': 'train'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            # t3_output [batch_size, num_actions]    action [batch_size, 1]
            t3_loss = self.t3.loss(output['t3_output'], data['action'])
            return {'t3_loss': t3_loss}
            
        elif method == 'test':
            common_output = {'method': 'test'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            return_output = {}
            # t3_output [batch_size, num_actions] --> [batch_size]
            return_output['t3_output'] = output['t3_output'].argmax(dim=-1, keepdim=False)
            return return_output





