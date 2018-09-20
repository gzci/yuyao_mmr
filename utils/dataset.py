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
        return False
    else:
        alternatives = x['alternatives'].replace("无法", "")
        if len(re.findall(r"不|对|没|假|无|否", alternatives)) > 0:
            return True
        else:
            return False


def parse_function(x):
    passage, query, answer = x["passage"].strip(), x["query"].strip(), x["answer"]
    passage = list(map(lambda w: w.word, pseg.cut(passage))) #w 有 flag 和word 这是jieba分词的结果
    query = list(map(lambda w: w.word, pseg.cut(query)))
    answer_id = 0
    if "无法" in answer:
        answer_id = 1
    if "不" in answer or "没" in answer or "假" in answer or \
            "无" in answer or "否" in answer or "错" in answer:
        answer_id = 2
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query), answer_id


def gen(filename):
    with open(filename, encoding="utf-8") as jf:
        jf = list(map(lambda x: ujson.decode(x), jf.readlines()))
        jf = list(filter(is_valid_data, jf))
        pool = ThreadPool(10)
        jf = list(pool.map(parse_function, jf))
        pool.close()
        pool.join()
        return jf


class MRC_Dataset(data.Dataset):

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


# data = MRC_Dataset(is_trainset=False)
# print(data[1])

