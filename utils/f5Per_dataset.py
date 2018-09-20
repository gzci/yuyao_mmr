import re
import sys
import math
import ujson
import torch
sys.path.append("..")
import jieba.posseg as pseg
import torch.utils.data as data
from utils.vocab import VocabDict
from multiprocessing.dummy import Pool as ThreadPool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID = "./mrc_data/ai_challenger_oqmrc_validationset.json"
TRAIN = "./mrc_data/ai_challenger_oqmrc_trainingset.json"
VOCAB = VocabDict()


def is_valid_data(x):
    if len(x["passage"]) >= 500:
        return True
    else:
        alternatives = x['alternatives'].replace("无法", "")
        if len(re.findall(r"不|对|没|假|无|否", alternatives)) > 0:
            return  False
        else:
            return True


def parse_function(x):
    passage, query, answer,al = x["passage"].strip(), x["query"].strip(), x["answer"] ,x["alternatives"]
    passage = list(map(lambda w: w.word, pseg.cut(passage))) #w 有 flag 和word 这是jieba分词的结果
    query = list(map(lambda w: w.word, pseg.cut(query)))
    al = list(map(lambda w: w.word, pseg.cut(al)))
    # print(passage,al,answer)
    # print(VOCAB.convert2idx(passage),VOCAB.convert2idx(query),VOCAB.get_idx(answer))
    if not VOCAB.convert2idx(passage).__contains__(VOCAB.get_idx(answer)):
        print("找不到跳过了",passage,answer,VOCAB.get_idx(answer))
        answer_id=9999
    else:
        answer_id =VOCAB.convert2idx(passage).index(VOCAB.get_idx(answer))
    #找到答案对应的idx 记得过滤
    # print(VOCAB.convert2idx(passage),answer_id)
    # print("******************")
    # print(passage,answer)
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query), answer_id

def is_in_passage(x):
    p = x["passage"]
    al = str(x["alternatives"]).strip().split("|")[:-1]
    if (al[0] in p) and (al[1] in p):
        return True
    else:
        return  False
def gen(filename):
    with open(filename, encoding="utf-8") as jf:
        jf = list(map(lambda x: ujson.decode(x), jf.readlines()))
        jf = list(filter(is_valid_data, jf))
        print(len(jf))
        jf = list(filter(is_in_passage, jf))
        # print(jf)
        print(len(jf))
        pool = ThreadPool(10)
        jf = list(pool.map(parse_function, jf))
        pool.close()
        pool.join()
        return jf


class f5per_Dataset(data.Dataset):

    def __init__(self, batch_size=64, is_trainset=True):
        if is_trainset:
            filename = TRAIN
        else:
            filename = VALID
        self.items = gen(filename)
        self.batch_size = batch_size

    def shuffle(self):
        self.items = [self.items[i] for i in torch.randperm(len(self.items))]

    def _pad(self, raw_data, feature=False):
        lengths = [len(x) for x in raw_data]
        max_length = max(lengths)
        pad_data = torch.zeros(len(raw_data), max_length)
        for i in range(len(raw_data)):
            data_length = lengths[i]
            pad_data[i].narrow(0, 0, data_length).copy_(torch.Tensor(raw_data[i]))
        if feature is True:
            return pad_data.float().to(device), torch.Tensor(lengths).float().to(device)
        else:
            return pad_data.long(), torch.Tensor(lengths).long()

    def _mask(self, seq_lens):
        mask = torch.zeros(len(seq_lens), torch.max(seq_lens))
        for i, seq_len in enumerate(seq_lens):
            mask[i][:seq_len] = 1
        return mask.float().to(device)

    def __getitem__(self, index):
        batch_items = self.items[index*self.batch_size:(index+1)*self.batch_size]
        docs, qrys, aws = [], [], []
        for d, q, a in batch_items:
            docs.append(d)
            qrys.append(q)
            aws.append(a)
        doc_pad, doc_lens = self._pad(docs)
        qry_pad, qry_lens = self._pad(qrys)
        doc_mask = self._mask(doc_lens)
        qry_mask = self._mask(qry_lens)
        aws = torch.Tensor(aws).long().to(device)
        return doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, aws

    def __len__(self):
        return math.ceil(len(self.items) / self.batch_size)


