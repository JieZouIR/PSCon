import torch.nn as nn
import torch

class PSCon_T4Model(nn.Module):
    def __init__(self, t4, response_len=50):
        super(PSCon_T4Model, self).__init__()
        self.response_len = response_len
        self.t4 = t4

    def query(self, t4_output):
        return self.t4.label(t4_output)

    def do_forward(self, context, query, passage, response, common_output):
        if 't4_output' not in common_output:
            self.t4(context, query, common_output)
        return common_output

    def forward(self, data, method):
        if method=='train':
            common_output = {'selected_query': data['selected_query'], 'selected_passage': data['selected_passage'],
                             'method': 'train'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            # t4_output [batch_size, num_queries]    selected_query [batch_size, queries]    query_loss_mask [batch_size, 1]
            t4_loss = self.t4.loss(output['t4_output'], data['selected_query'], data['query_loss_mask'])
            return {'t4_loss': t4_loss}
            
        elif method == 'test':
            common_output = {'method': 'test'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            return_output = {}
            # t4_output [batch_size, num_queries]
            return_output['t4_output'] = output['t4_output']
            return return_output





