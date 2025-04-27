import torch.nn as nn
import torch
import copy

class PSCon_T2Model(nn.Module):
    def __init__(self, t2, response_len=50):
        super(PSCon_T2Model, self).__init__()
        self.response_len = response_len
        self.t2 = t2

    def state(self, t2_output):
        return self.t2.label(t2_output)

    def do_forward(self, context, query, passage, response, common_output):
        if 't2_output' not in common_output:
            self.t2(context, common_output)
        return common_output

    def forward(self, data, method):
        if method=='train':
            common_output = {'selected_query': data['selected_query'], 'selected_passage': data['selected_passage'],
                             'method': 'train'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            t2_loss = self.t2.loss(output['t2_output'], data['state'], data['state_loss_mask'])
            return {'t2_loss': t2_loss}
            
        elif method == 'test':
            common_output = {'method': 'test'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            return_output = {}
            # t2_output [batch_size, context_len]
            return_output['t2_output'] = copy.deepcopy(data['context']).masked_fill_(torch.sigmoid(output['t2_output']) < 0.5, 0)
            return return_output





