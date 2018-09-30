import os
import re
import sys
import math
import ujson
import torch
sys.path.append("..")
import jieba.posseg as pseg
import torch.utils.data as data
from utils.vocab import VocabDict
from multiprocessing import Pool

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB = VocabDict()
invalid_items = []


def read_json(filename):
    with open(filename, encoding="utf-8") as jf:
        jf = list(map(lambda x: ujson.decode(x), jf.readlines()))
    return jf

def is_valid_data(x):
    if len(x["passage"]) > 500:
        invalid_items.append(x)
        return False
    else:
        al = list(filter(lambda c: len(c.strip()) > 0, str(x['alternatives']).split("|")))
        if len(al) < 3:
            invalid_items.append(x)
            return False
        else:
            return True


def parse_function(x):
    passage, query, alternatives, query_id = x["passage"].strip(), x["query"].strip(), \
                                             x["alternatives"].strip(), x['query_id']
    passage = list(map(lambda w: w.word, pseg.cut(passage)))
    query = list(map(lambda w: w.word, pseg.cut(query)))
    # positive: 0    unknown: 1   negative: 2
    alternatives = str(alternatives).split("|")
    alternatives= list(map(lambda w:w.strip(),alternatives))
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query), alternatives, query_id



def gen(filename):
    jf = read_json(filename)
    jf = list(filter(is_valid_data, jf))
    pool = Pool(10)
    jf = list(pool.map(parse_function, jf))
    pool.close()
    pool.join()
    return jf


class Pred_Dataset922(data.Dataset):

    def __init__(self, filename, batch_size):
        self.items = gen(filename)
        self.batch_size = batch_size

    @staticmethod
    def _pad(raw_data, feature=False):
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

    @staticmethod
    def _mask(seq_lens):
        mask = torch.zeros(len(seq_lens), torch.max(seq_lens))
        for i, seq_len in enumerate(seq_lens):
            mask[i][:seq_len] = 1
        return mask.float().to(device)

    def __getitem__(self, index):
        batch_items = self.items[index * self.batch_size:(index + 1) * self.batch_size]
        docs, qrys, alternatives, qry_ids = [], [], [], []
        for d, q, al, idx in batch_items:
            docs.append(d)
            qrys.append(q)
            alternatives.append(al)
            qry_ids.append(idx)
        doc_pad, doc_lens = self._pad(docs)
        qry_pad, qry_lens = self._pad(qrys)
        doc_mask = self._mask(doc_lens)
        qry_mask = self._mask(qry_lens)
        return doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, alternatives, qry_ids

    def __len__(self):
        print(len(self.items))
        return math.ceil(len(self.items) / self.batch_size)