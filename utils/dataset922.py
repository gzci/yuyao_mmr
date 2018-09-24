import re
import sys
import math
import ujson
import torch
import ujson
import jieba
import pickle
from multiprocessing import Pool
sys.path.append("..")
import jieba.posseg as pseg
import torch.utils.data as data
from utils.vocab import VocabDict
from multiprocessing.dummy import Pool as ThreadPool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VALID = "../mrc_data/newvaild.json"
TRAIN = "../mrc_data/newtrain.json"
VOCAB = VocabDict()

#这个是 Q size 这么大小的{0，1}数组

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


def is_valid_data(x):
    if len(x['passage']) > 500:
        return False
    al = list(filter(lambda c: len(c.strip()) > 0, str(x['alternatives']).split("|")))
    al = list(map(lambda c: c.strip(), al))
    if len(al) < 3 or x['answer'].strip() not in al:
        return False
    else:
        return True


def parse_function(x):
    passage, query, answer, al = x["passage"].strip(), x["query"].strip(), x["answer"].strip(), x["alternatives"]
    passage = list(map(lambda w: w.word, pseg.cut(passage)))
    query = list(map(lambda w: w.word, pseg.cut(query)))
    al = list(map(lambda c: c.strip(), al.split("|")))
    answer_id = al.index(answer)
    al = list(map(lambda c: VOCAB.convert2idx(list(jieba.cut(c))), al))
    # w_ms, q_w_ms = word_match(passage, query)
    # c_ms, q_c_ms = char_match(passage, query)
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query), al, answer_id
        # w_ms, c_ms, q_w_ms, q_c_ms


def gen(filename):
    with open(filename, 'r', encoding='utf-8')as f:
        jf = list(ujson.load(f))
        jf = list(filter(is_valid_data, jf))
        pool = Pool(4)
        jf = list(pool.map(parse_function, jf))
        pool.close()
        pool.join()
        # pickle.dump(jf, open("valid.pkl", "wb"))
        return jf

class Dataset922(data.Dataset):

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