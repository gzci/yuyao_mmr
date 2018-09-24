import ujson
import jieba
import pickle
import jieba.posseg as pseg
from multiprocessing import Pool
from utils.vocab import VocabDict


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
    w_ms, q_w_ms = word_match(passage, query)
    c_ms, q_c_ms = char_match(passage, query)
    print(al)
    return VOCAB.convert2idx(passage), VOCAB.convert2idx(query), al, answer_id, \
        w_ms, c_ms, q_w_ms, q_c_ms


def gen(filename):
    with open(filename, 'r', encoding='utf-8')as f:
        jf = list(ujson.load(f))
        jf = list(filter(is_valid_data, jf))
        pool = Pool(4)
        jf = list(pool.map(parse_function, jf))
        pool.close()
        pool.join()
        print(len(jf))
        pickle.dump(jf, open("valid.pkl", "wb"))
        return jf


if __name__ == '__main__':
    print(gen(VALID))