import  torch as t
import torch.autograd.variable as Variable
import  torch.nn as nn
import jieba.posseg as pseg
from utils.vocab import *
from utils.f5Per_dataset import *
# rnn = nn.GRU(10, 20, 2)
# input = Variable(t.randn(5, 3, 10))
# h0 = Variable(t.randn(4, 3, 20))
# output, hn = rnn(input, None)
# print(output.size())
# x= t.argmax(t.randn(5,1,3).squeeze(1),1)
# y=t.argmax(t.randn(5,1,3).squeeze(1),1)
# print(x)
# print(y)
# print((x==y).sum().item())
#
# te="无法确定"
# print(len(te.split("|")[:-1]))
# pad_data = t.zeros(3, 5)
# raw_data=t.ones(3,3)
# print(pad_data)
# pad_data[0].narrow(0, 0, 3).copy_(t.Tensor(raw_data[0]))
# print(pad_data)
# VOCAB = VocabDict()
# print(VOCAB.get_idx("法律"))
# print(VOCAB.get_word(26464))

import re


# print(re.findall(r"不|定", '不一定|一定|确定'))
# print(len(re.findall(r"不|定", '不一定|一定|确定')))
data = f5per_Dataset(is_trainset=False)
# print(data[1])
