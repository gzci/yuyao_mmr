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


def word_match(c, q):
    c_match = [int(w in q) for w in c]
    q_match = [int(w in c) for w in q]
    return c_match, q_match


def char_match(c, q):
    _c = "".join(c)
    _q = "".join(q)
    char_c_match, char_q_match = [], []
    for w in c:
        appear_times = 0
        w_len = len(w)
        if w_len > 0:
            for w_c in w:
                if w_c in _q:
                    appear_times += 1
            score = appear_times / w_len
        else:
            score = 0
        char_c_match.append(score)
    for w in q:
        w_len = len(w)
        appear_times = 0
        if w_len > 0:
            for w_c in w:
                if w_c in _c:
                    appear_times += 1
            score = appear_times / w_len
        else:
            score = 0
        char_q_match.append(score)
    return char_c_match, char_q_match


def read_json(filename):
    with open(filename, encoding="utf-8") as jf:
        jf = list(map(lambda x: ujson.decode(x), jf.readlines()))
    return jf


def is_valid_data(x):
    if len(x["passage"]) >= 500:
        invalid_items.append(x)
        return False
    else:
        alternatives = x['alternatives'].replace("无法", "")
        if len(re.findall(r"不|对|没|假|无|否", alternatives)) > 0:
            return True
        else:
            invalid_items.append(x)
            return False


def parse_function(x):
    passage, query, alternatives, query_id = x["passage"].strip(), x["query"].strip(), \
                                             x["alternatives"].strip(), x['query_id']
    passage = list(map(lambda w: w.word, pseg.cut(passage)))
    query = list(map(lambda w: w.word, pseg.cut(query)))
    # positive: 0    unknown: 1   negative: 2
    alternatives = str(alternatives).split("|")
    w_ms, q_w_ms = word_match(passage, query)
    print(w_ms)
    print(q_w_ms)
    c_ms, q_c_ms = char_match(passage, query)
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query), alternatives, query_id, \
        w_ms, c_ms, q_w_ms, q_c_ms


def gen(filename):
    jf = read_json(filename)
    jf = list(filter(is_valid_data, jf))
    pool = Pool(10)
    jf = list(pool.map(parse_function, jf))
    pool.close()
    pool.join()
    return jf


class Pred_Dataset(data.Dataset):

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
        docs, qrys, alternatives, qry_ids, w_ms, c_ms, q_w_ms, q_c_ms = [], [], [], [], [], [], [], []
        for d, q, al, idx, w_m, c_m, q_w_m, q_c_m in batch_items:
            docs.append(d)
            qrys.append(q)
            alternatives.append(al)
            qry_ids.append(idx)
            w_ms.append(w_m)
            c_ms.append(c_m)
            q_w_ms.append(q_w_m)
            q_c_ms.append(q_c_m)
        doc_pad, doc_lens = self._pad(docs)
        qry_pad, qry_lens = self._pad(qrys)
        doc_mask = self._mask(doc_lens)
        qry_mask = self._mask(qry_lens)
        w_ms, _ = self._pad(w_ms, feature=True)
        c_ms, _ = self._pad(c_ms, feature=True)
        q_w_ms, _ = self._pad(q_w_ms, feature=True)
        q_c_ms, _ = self._pad(q_c_ms, feature=True)
        return doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, alternatives, qry_ids, \
            w_ms, c_ms, q_w_ms, q_c_ms

    def __len__(self):
        return math.ceil(len(self.items) / self.batch_size)