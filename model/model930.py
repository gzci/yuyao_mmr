import torch.nn.functional as F
from model.attn import *
from model.layers import VariableLengthGRU


def mean(ipt, lens, dim=2):
    attn = torch.sum(ipt, dim=dim, keepdim=True)
    lens = lens.unsqueeze(1).unsqueeze(2).expand_as(attn)
    averaged_attention = attn / lens.float().to(device)
    return averaged_attention


class Model930(nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(Model930, self).__init__()

        self.drop = nn.Dropout(0.15)
        self.qry_encode = VariableLengthGRU(input_size=embed_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)
        self.doc_encode = VariableLengthGRU(input_size=embed_dim ,
                                            hidden_size=hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)
        self.gru_l1=VariableLengthGRU(input_size=8*hidden_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)

        self.gru_l2 =VariableLengthGRU(input_size= 16*hidden_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)

        self.gru_out=VariableLengthGRU(input_size=24*hidden_dim,
                                            hidden_size=4*hidden_dim,
                                            num_layers=2,
                                            batch_first=True,
                                            bidirectional=True)

        self.qry_self_attention = Attention(input_size=2 * hidden_dim,
                                            output_size=2 * hidden_dim)

        self.doc_self_attention = Attention(input_size=2 * hidden_dim,
                                            output_size=2 * hidden_dim)
        self.attn_l1 = Attention(input_size=2 * hidden_dim,
                                            output_size=2 * hidden_dim)
        self.attn_l2 = Attention(input_size=2 * hidden_dim,
                                 output_size=2 * hidden_dim)
        self.line_layer1 = nn.Linear(8*hidden_dim, 2*hidden_dim)

        # self.line_layer2=nn.Linear(4*hidden_dim,hidden_dim)

        self.out_layer = nn.Linear(2*hidden_dim, 3)

    def init_weights(self):
        for weight in self.dense_layer1.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)
        for weight in self.out_layer.parameters():
            if len(weight.size()) > 1:
                init.xavier_normal_(weight.data)

    def forward(self, doc_embed, doc_len, doc_mask, qry_embed, qry_len, qry_mask):

       #doc_embed = self.drop(doc_embed)
       #qry_embed = self.drop(qry_embed)
        #  64,Qlen,200->
        qry_encode = self.qry_encode(qry_embed, qry_len) #64,Qlen,128*2
        qry_output, qry_attn = self.qry_self_attention(qry_encode, qry_mask,
                                                       qry_encode, qry_mask,
                                                       qry_encode, qry_mask)
        #64,Dlen,128*2
        doc_encode = self.doc_encode(doc_embed, doc_len)
        doc_output, doc_attn = self.doc_self_attention(doc_encode, doc_mask,
                                                       doc_encode, doc_mask,
                                                       doc_encode, doc_mask)

        #                           64,Qlen,128*2
        qry_output=mean(torch.cat([qry_output, qry_encode], dim=2), qry_len, dim=1)\
            .expand(doc_embed.size(0), doc_embed.size(1), 2*qry_output.size(-1))
        # 64，Dlen,4*128

        #64，Dlen,8*128
        catl1=torch.cat([qry_output,doc_encode,doc_output],dim=2)

        gru_l1_encode=self.gru_l1(catl1,doc_len)
        # 64,Dlen,128*2
        attn_l1_output, attn_l1_attn = self.doc_self_attention(gru_l1_encode, doc_mask,
                                                               gru_l1_encode, doc_mask,
                                                               gru_l1_encode, doc_mask)
        # 64,Dlen,128*2
        catl2=torch.cat([attn_l1_output, qry_output,gru_l1_encode,catl1], dim=2)
        #64,Dlen,16*128

        gru_l2_encode=self.gru_l2(catl2,doc_len)
        #64,Dlen,2*128
        attn_l2_output, attn_l2_attn = self.doc_self_attention(gru_l2_encode, doc_mask,
                                                               gru_l2_encode, doc_mask,
                                                               gru_l2_encode, doc_mask)
        # 64,Dlen,2*128

        catl3=torch.cat([attn_l2_output, qry_output,gru_l2_encode,catl2], dim=2)
        #64，Dlen,24*128
        gru_out_encode=self.gru_out(catl3,doc_len)
        # 64，Dlen,2*128

        gru_out_encode = mean(gru_out_encode, doc_len, dim=1)
        out = self.line_layer1(gru_out_encode)
        out = F.relu(out)
        # out = self.line_layer2(out)
        # out= F.sigmoid(out)
        out = self.out_layer(out).squeeze(1)

        return out


