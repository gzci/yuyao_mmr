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
import os

# print(open("./mrc_data/valid.pkl"))
# def mean(ipt, lens, dim=2):
#     attn = torch.sum(ipt, dim=dim, keepdim=True)
#     print(attn)
#     lens = lens.unsqueeze(1).unsqueeze(2).expand_as(attn)
#     print(lens)
#     averaged_attention = attn / lens.float()
#     return averaged_attention
# TRAINkpl = "./mrc_data/train.pkl"
# VALIDkpl = "./mrc_data/valid.pkl"
# from utils.dataset922 import Dataset922
# if os.path.exists(VALIDkpl):
#     print(1)
# else: print(2)
from utils.dataset922 import Dataset922

data = Dataset922(is_trainset=False)
print(data[1])
#
# a = torch.Tensor(torch.randn(64)).reshape([4,2,-1])
# print(a.size())
# torch.transpose(a, 1, 2)
# print(a.size())
# l =torch.ones(4).float()
# torch.sum(a, dim=1)
# print(a)
# a=mean(a,l,dim=1)
# print(a)
# print(a.size())



# # embed_mat = np.load('./mrc_data/vectors.npy')
# embed = nn.Embedding.from_pretrained(torch.Tensor(embed_mat))
# embed = nn.Embedding.from_pretrained(torch.randn([1000, 5]))
# al=[[331,123],[41, 331],[650, 673]]
# ax=t.arange(6)
# print(ax.reshape(2,3))
# print(al.reshape(1,-1))


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