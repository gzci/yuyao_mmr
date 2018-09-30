import torch.nn.functional as F
from model.attn import *
from model.layers import VariableLengthGRU


def mean(ipt, lens, dim=2):
    attn = torch.sum(ipt, dim=dim, keepdim=True)
    lens = lens.unsqueeze(1).unsqueeze(2).expand_as(attn)
    averaged_attention = attn / lens.float().to(device)
    return averaged_attention


class Model922(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(Model922, self).__init__()

        self.drop = nn.Dropout(0.15)
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


        self.line_layer1 = nn.Linear(8*hidden_dim, 4*hidden_dim)

        self.line_layer2=nn.Linear(4*hidden_dim,hidden_dim)

        self.out_layer = nn.Linear(hidden_dim, 3)

    def init_weights(self):
        for weight in self.dense_layer1.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)
        for weight in self.out_layer.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)

    def forward(self, doc_embed, doc_len, doc_mask, qry_embed, qry_len, qry_mask):

        doc_embed = self.drop(doc_embed)
        qry_embed = self.drop(qry_embed)
        #  64,Qlen,200->
        qry_encode = self.qry_encode(qry_embed, qry_len) #64,Qlen,128*2
        qry_output, qry_attn = self.qry_self_attention(qry_encode, qry_mask,
                                                       qry_encode, qry_mask,
                                                       qry_encode, qry_mask)
        # 64,Dlen,128*4
        qry_output = mean(torch.cat([qry_output, qry_encode], dim=2), qry_len, dim=1)\
            .expand(doc_embed.size(0), doc_embed.size(1), 2 * qry_output.size(-1))
        #64,Dlen,128*4+200
        doc_embed = torch.cat([doc_embed, qry_output], dim=2)
        doc_encode = self.doc_encode(doc_embed, doc_len)
        #64,Dlen,128*2
        doc_output, doc_attn = self.doc_self_attention(doc_encode, doc_mask,
                                                       doc_encode, doc_mask,
                                                       doc_encode, doc_mask)
        doc_output = mean(torch.cat([doc_output, doc_encode, qry_output], dim=2), doc_len, dim=1)

        out = self.line_layer1(doc_output)
        out = F.relu(out)
        out = self.line_layer2(out)
        out= F.sigmoid(out)
        out = self.out_layer(out).squeeze(1)

        return out


