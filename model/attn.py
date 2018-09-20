# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init

_eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_mask(input, mask, axis=1, epsilon=1e-12):

    shift, _ = torch.max(input, axis, keepdim=True)

    shift = shift.expand_as(input).to(device)

    target_exp = torch.exp(input - shift) * mask

    normalize = torch.sum(target_exp, axis, keepdim=True).expand_as(target_exp)

    softm = target_exp / (normalize + epsilon)

    return softm.to(device)


class AttnOverAttn(nn.Module):
    
    def __init__(self, mode='dot'):
        super(AttnOverAttn, self).__init__()
        self.mode = mode
        
    def forward(self, doc, doc_mask, qry, qry_mask):
        
        d_mask = doc_mask.unsqueeze(2)
        q_mask = qry_mask.unsqueeze(2)
        dot_mask = torch.bmm(d_mask, torch.transpose(q_mask, 1, 2))
        if self.mode == 'dot':
            c_q_dot = torch.bmm(doc, torch.transpose(qry, 1, 2))
        if self.mode == 'cos':
            dot_xy = torch.bmm(doc, torch.transpose(qry, 1, 2))
            xx = torch.sqrt(torch.sum(doc * doc, 2)).unsqueeze(1)
            yy = torch.sqrt(torch.sum(qry * qry, 2)).unsqueeze(1)
            c_q_dot = dot_xy / (torch.bmm(torch.transpose(xx, 1, 2), yy) + _eps)
        
        # column-wise soft_max
        # query to context attention
        qtc_attention = softmax_mask(c_q_dot, dot_mask, axis=1)

        # row-wise soft_max
        # context to query attention
        ctq_attention = softmax_mask(c_q_dot, dot_mask, axis=2)
        
        attention = qtc_attention * ctq_attention
        
        c_q_align = torch.bmm(attention, qry)
        
        return attention, c_q_align


class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.1):
        super(DotProductAttention, self).__init__()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        attn = torch.bmm(q, torch.transpose(k, 1, 2))

        col_attn = softmax_mask(attn, mask, axis=1)
        row_attn = softmax_mask(attn, mask, axis=2)

        attn = col_attn * row_attn
        # attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class Attention(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Attention, self).__init__()
        self.w_qs = nn.Linear(input_size, output_size)
        self.w_ks = nn.Linear(input_size, output_size)
        self.w_vs = nn.Linear(input_size, output_size)
        self.attn = DotProductAttention()
        self.init_weights()
        
    def init_weights(self):
        for weight in self.w_qs.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)
        for weight in self.w_ks.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)
        for weight in self.w_vs.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)
        
    def forward(self, q, q_mask, k, k_mask, v, v_mask):

        q, k, v = self.w_qs(q), self.w_vs(k), self.w_vs(v)

        d_mask = q_mask.unsqueeze(2)
        q_mask = k_mask.unsqueeze(2)
        dot_mask = torch.bmm(d_mask, torch.transpose(q_mask, 1, 2))

        output, attn = self.attn(q, k, v, mask=dot_mask)

        return output, attn
        

