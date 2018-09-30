import os
import logging
import random
import numpy as np
import torch.nn as nn
from model.model922 import Model922
from utils.pred_dataset922 import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_mat = np.load('./mrc_data/vectors.npy')
embed = nn.Embedding.from_pretrained(torch.Tensor(embed_mat))

# hyperparameter
embed_dim = 200
hidden_dim = 150
batch_size = 64
VALID_SET = "./mrc_data/ai_challenger_oqmrc_validationset.json"
TESTA_SET = "./mrc_data/ai_challenger_oqmrc_testa.json"
MODEL_PATH = './checkpoints/c_m_acc-0.725454_ep-43.pth'


def write_to_file(pred_res, filename):
    pred_res = sorted(pred_res.items(), key=lambda item: item[0])
    with open(filename, "w") as pred_file:
        for res in pred_res:
            idx, answer = res
            pred_file.write("%d\t%s\n" % (idx, answer))


def random_guess(pred_res, filename, name=TESTA_SET):
    logger.info(str(len(invalid_items)))
    for it in invalid_items:
        p= it["passage"]
        al = str(it["alternatives"]).strip().split("|")[:-1]

        if len(al) < 2:
            pred_res[it["query_id"]] = str(it["alternatives"]).strip().split("|")[0]
        else:
            if (al[0] in p) and (al[1] not in p):
                pred_res[it["query_id"]] = al[0]
            elif (al[0] not in p) and (al[1] in p):
                pred_res[it["query_id"]] = al[1]
            else:
                pred_res[it["query_id"]] = str(it["alternatives"]).strip().split("|")[0]
    logger.info(str(len(pred_res)))
    write_to_file(pred_res, filename)


def transform(pred_, alternatives):
    res = []
    for i in range(len(pred_)):
        p_label = pred_[i]
        al = alternatives[i]
        if p_label == 2:
            for a in al:
                if "无法" in a:
                    res.append(a)
                    break
        elif p_label == 1:
            for a in al:
                if len(re.findall(r"不|错|没|假|无|否", a)) > 0:
                    res.append(a)
                    break
        else:
            not_found = True
            for a in al:
                if len(re.findall(r"不|错|没|假|无|否", a)) == 0 and "无法" not in a:
                    res.append(a)
                    not_found = False
                    break
            if not_found:
                res.append(random.randint(0,1))
    return res


def pred(model, dataset, name=TESTA_SET):

    pred_res = {}

    for ii in range(dataset.__len__()):
        doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, alternatives, qry_ids= dataset[ii]
        doc_pad = embed(doc_pad).to(device)
        qry_pad = embed(qry_pad).to(device)
        pred_ = model(doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask)
        pred_ = torch.argmax(pred_, 1).cpu().tolist()
        pred_ = transform(pred_, alternatives)
        for idx, answer in zip(qry_ids, pred_):
            pred_res[idx] = answer
            logger.info("%d:%s" % (idx, answer))

    if (len(pred_res) == 10000 and name == TESTA_SET) or (len(pred_res) == 30000 and name == VALID_SET):
        write_to_file(pred_res, "rnn_prediction.txt")
    else:
        logger.info(str(len(pred_res)))
        random_guess(pred_res, "rnn_prediction.txt", name=name)


if __name__ == '__main__':

    logger.info("Start ......")

    logger.info("Load model ......")
    c_model = Model922(embed_dim, hidden_dim)
    c_model.load_state_dict(torch.load(MODEL_PATH))
    c_model.to(device)
    c_model.eval()
    print("===================================================")
    print(c_model)
    print("===================================================")

    logger.info("Load prediction set ......")
    pred_dataset = Pred_Dataset922(filename=TESTA_SET, batch_size=batch_size)

    logger.info("Start predicate ......")
    pred(c_model, pred_dataset, TESTA_SET)
#0.741548