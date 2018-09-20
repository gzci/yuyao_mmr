import sys
import pickle
sys.path.append("..")
from utils import constants


class VocabDict:

    def __init__(self, dict_path='./mrc_data/dict.pkl'):
        self.idx2word = self._load(dict_path)
        self.word2idx = {word: idx for idx, word in self.idx2word.items()}

    @staticmethod
    def _load(dict_path):
        with open(dict_path, 'rb') as dt:
            dct = pickle.load(dt)
        return dct

    def get_idx(self, word):
        return self.word2idx.get(word, constants.UNK)

    def get_word(self, idx):
        return self.idx2word.get(idx, constants.UNK_WORD)

    def convert2idx(self, words):
        vec = list(map(lambda w: self.get_idx(w), words))
        return vec

    def convert2word(self, ids):
        vec = list(map(lambda w: self.get_word(w), ids))
        return vec

    def size(self):
        return len(self.idx2word)

vab=VocabDict()
# print(len(vab.idx2word))
# print(len(vab.idx2word))
# print(vab.get_idx('你是傻逼'))
# print(vab.get_word(507056))
# print(vab.convert2word([1,2,3,4,5]))       ['的', '。', '、', '在', '是']

#
# for i ,j in vab.idx2word.items():
#     print(i,j)






