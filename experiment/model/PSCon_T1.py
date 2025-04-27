import torch.nn as nn
import torch
import copy

class PSCon_T1Model(nn.Module):
    def __init__(self, t1, response_len=50):
        super(PSCon_T1Model, self).__init__()
        self.response_len = response_len
        self.t1 = t1

    def intent(self, t1_output):
        return self.t1.label(t1_output)

    def do_forward(self, context, query, passage, response, common_output):
        if 't1_output' not in common_output:
            self.t1(context, common_output)
        return common_output

    def forward(self, data, method):
        if method=='train':
            common_output = {'selected_query': data['selected_query'], 'selected_passage': data['selected_passage'],
                             'method': 'train'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            t1_loss = self.t1.loss(output['t1_output'], data['intent'])
            return {'t1_loss':t1_loss}
            
        elif method == 'test':
            common_output = {'method': 'test'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            return_output = {}
            # t1_output [batch_size, num_intents] --> [batch_size]
            return_output['t1_output'] = output['t1_output'].argmax(dim=-1, keepdim=False)
            return return_output





