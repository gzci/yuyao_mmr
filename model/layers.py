# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sort_batch(data, seq_len):

    sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)

    sorted_data = data[sorted_idx.data]

    _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)

    return sorted_data.to(device), sorted_seq_len.to(device), reverse_idx.to(device)


class VariableLengthGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers=1, drop_rate=0.0, batch_first=True, bidirectional=True):
        super(VariableLengthGRU, self).__init__()
        
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=bidirectional,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          dropout=drop_rate)
        self.init_weights()

    def init_weights(self):
        for weight in self.gru.parameters():
            if len(weight.size()) > 1:
                init.orthogonal_(weight.data)
        
    def forward(self, x, x_lens):
        # sort
        s_x, s_x_len, reverse_x_idx = sort_batch(x, x_lens)
        
        embeds = pack(s_x, list(s_x_len.data), batch_first=True)
        
        # encode
        out, _ = self.gru(embeds, None)
        
        # unpack
        out, _ = unpack(out, batch_first=True)
        
        # resort
        out = out[reverse_x_idx.data]
        
        return out
