import torch as t
from torch import  nn
import torch
import visdom
import logging
import numpy as np
import torch.nn as nn
from torch.nn import utils
import torch.optim as optim
from model.models import ClassificationModel
from utils.dataset import MRC_Dataset
from torch.autograd import Variable as V
def mean(ipt, lens, dim=2):
    attn = torch.sum(ipt, dim=dim, keepdim=True)
    lens = lens.unsqueeze(1).unsqueeze(2).expand_as(attn)
    averaged_attention = attn / lens.float()
    return averaged_attention
# # embed_mat = np.load('./mrc_data/vectors.npy')
# embed = nn.Embedding.from_pretrained(torch.Tensor(embed_mat))
# embed = nn.Embedding.from_pretrained(torch.randn([1000, 5]))
al=[[331,123],[41, 331],[650, 673]]
ax=t.arange(6)
print(ax.reshape(2,3))
print(al.reshape(1,-1))


# al =t.Tensor(al)
# lens=t.Tensor([a.size() for a in al ])
# print()
# al=mean(al,lens,1)

# al.squeeze(0)
# print(al.size())
# print(al)
# do = V(t.LongTensor(al))
# print((embed(do)))
# print((embed(do).size()))
# print(type(t.arange(3,0,-1)))


# an Embedding module containing 10 tensors of size 3
# embedding = nn.Embedding(11, 3)
# # a batch of 2 samples of 4 indices each
# input = (torch.LongTensor([[1,2,4,5,8],[4,3,2,9,8],[10,3,2,9,8]]))
# print(embedding(input))










# s  =  V(t.randn(1,2))
# l = V(t.Tensor([1])).long()
#
# pre = nn.CrossEntropyLoss()
#
# loss = pre(s,l)
#
# print(s)
# print(loss)
#
# q=t.Tensor(t.randn(2,5))
# print(q)
#
# q=q.unsqueeze(2)
# q=q.unsqueeze(3)
# print\
#     (q.size())