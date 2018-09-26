import torch.nn.functional as F
from model.attn import *
from model.layers import VariableLengthGRU


def mean(ipt, lens, dim=2):
    attn = torch.sum(ipt, dim=dim, keepdim=True)
    lens = lens.unsqueeze(1).unsqueeze(2).expand_as(attn)
    averaged_attention = attn / lens.float().to(device)
    return averaged_attention


class ClassificationModel(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(ClassificationModel, self).__init__()

        # self.drop = nn.Dropout(0.1)

        self.qry_encode = VariableLengthGRU(input_size=embed_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)

        self.qry_self_attention = Attention(input_size=2 * hidden_dim,
                                            output_size=2 * hidden_dim)

        self.doc_encode = VariableLengthGRU(input_size=embed_dim+4*hidden_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)

        self.doc_self_attention = Attention(input_size=2 * hidden_dim,
                                            output_size=2 * hidden_dim)

        self.dense_layer1 = nn.Linear(8*hidden_dim, hidden_dim)

        self.out_layer = nn.Linear(hidden_dim, 3)

    def init_weights(self):
        for weight in self.dense_layer1.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)
        for weight in self.out_layer.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)

    def forward(self, doc_embed, doc_len, doc_mask, qry_embed, qry_len, qry_mask):

        # doc_embed = self.drop(doc_embed)
        # qry_embed = self.drop(qry_embed)

        qry_encode = self.qry_encode(qry_embed, qry_len)    # batch * qry_len * (2*hidden_dim)
        qry_output, qry_attn = self.qry_self_attention(qry_encode, qry_mask,
                                                       qry_encode, qry_mask,
                                                       qry_encode, qry_mask)
        qry_output = mean(torch.cat([qry_output, qry_encode], dim=2), qry_len, dim=1)\
            .expand(doc_embed.size(0), doc_embed.size(1), 2 * qry_output.size(-1))    # batch * doc_len * (2*hidden_dim) 32,256,40
        # print('qry_output',qry_output.size())
        doc_embed = torch.cat([doc_embed, qry_output], dim=2)   # batch * doc_len * (embed_dim+hidden_dim)
        doc_encode = self.doc_encode(doc_embed, doc_len)    # batch * doc_len * (2*hidden_dim)
        doc_output, doc_attn = self.doc_self_attention(doc_encode, doc_mask,
                                                       doc_encode, doc_mask,
                                                       doc_encode, doc_mask)
        doc_output = mean(torch.cat([doc_output, doc_encode, qry_output], dim=2), doc_len, dim=1)

        out = self.dense_layer1(doc_output)
        out = F.relu(out)
        out = self.out_layer(out).squeeze(1)

        return out




'''

model 整个过程
qry :32,214,3 ->32,214,10 -> 32,214,10 (atten) -> 32,214,20 ->32,1,20(mean) ->32 ,256,20(expand)
doc:  32,256,3 ->32,256,23(cat) -> 32,256,10(bigru) ->32,256,10 (atten)
->32,256,40(cat) ->32,1,40(mean) ->32,1,5 ->32,1,3->32,3

'''
# mat = torch.randn((30, 20))
# qry = torch.Tensor([[1,2,3,4,5,6]]).long()
# doc = torch.Tensor([[1,2,3,4,5,6,7,8,9]]).long()
# qry_len = torch.Tensor([6]).long()
# doc_len = torch.Tensor([9]).long()
# qry_mask = torch.Tensor([[1,1,1,1,1,1]])
# doc_mask = torch.Tensor([[1,1,1,1,1,1,1,1,1]])
# target = torch.Tensor([1]).long()
# loss = nn.CrossEntropyLoss()
# model = ClassificationModel(mat, 20, 10)
# out = model(doc, doc_len, doc_mask, qry, qry_len,qry_mask)
# print(out)
# print(loss(out, target))

