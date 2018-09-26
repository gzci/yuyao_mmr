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

from multiprocessing.dummy import Pool as ThreadPool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.vocab import VocabDict

VALID = "./mrc_data/newvaild.json"
TRAIN = "./mrc_data/newtrain.json"
VOCAB = VocabDict()




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
    al = list(map(lambda x:x.strip(),al.split("|")))
    answer_id = al.index(answer)
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query),  answer_id


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
    def shuffle(self):
        self.items = [self.items[i] for i in torch.randperm(len(self.items))]

    def __init__(self, is_trainset,batch_size=64):
        if is_trainset:
            filename = TRAIN
        else:
            filename = VALID
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

if __name__ == '__main__':

    data = Dataset922(is_trainset=False)
    print(data[1])