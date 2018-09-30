import os
import logging
import random
import numpy as np
import torch.nn as nn
from model.models import ClassificationModel
from utils.pred_dataset import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameter
embed_dim = 200
hidden_dim = 128
batch_size = 128

VALID_SET = "./mrc_data/valid.json"
TESTA_SET = "./mrc_data/ai_challenger_oqmrc_testa.json"
MODEL_PATH = './checkpoints/' + 'model128_0.753.pth'


def write_to_file(pred_res, filename):
    pred_res = sorted(pred_res.items(), key=lambda item: item[0])
    with open(filename, "w") as pred_file:
        for res in pred_res:
            idx, answer = res
            pred_file.write("%d\t%s\n" % (idx, answer))


def random_guess(pred_res, filename, name=TESTA_SET):
    logger.info(str(len(invalid_items)))
    for it in invalid_items:
        print(it["query_id"], str(it["alternatives"]))
        als = str(it["alternatives"]).strip().split("|")
        if len(als[0].strip()) > 0:
            pred_res[it["query_id"]] = als[0].strip()
        else:
            pred_res[it["query_id"]] = als[1].strip()
    logger.info(str(len(pred_res)))
    write_to_file(pred_res, filename)


def transform(pred_, alternatives):
    res = []
    for i in range(len(pred_)):
        p_label = pred_[i]
        al = alternatives[i]
        res.append(al[p_label])
    return res


def pred(model, dataset, name=TESTA_SET):

    pred_res = {}

    for ii in range(dataset.__len__()):
        doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, als, qry_ids, w_ms, c_ms, q_w_ms, q_c_ms, alss = dataset[ii]
        pred_ = model(doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, w_ms, c_ms, q_w_ms, q_c_ms, als)
        pred_ = torch.argmax(pred_, 1).cpu().tolist()
        pred_ = transform(pred_, alss)
        for idx, answer in zip(qry_ids, pred_):
            pred_res[idx] = answer
            logger.info("%d\t%s" % (idx, answer))

    if (len(pred_res) == 10000 and name == TESTA_SET) or (len(pred_res) == 30000 and name == VALID_SET):
        write_to_file(pred_res, "single_prediction.txt")
    else:
        logger.info(str(len(pred_res)))
        random_guess(pred_res, "single_prediction.txt", name=name)


if __name__ == '__main__':

    logger.info("Start ......")

    logger.info("Load model ......")
    c_model = ClassificationModel(embed_dim, hidden_dim)
    c_model.load_state_dict(torch.load(MODEL_PATH))
    c_model.to(device)
    c_model.eval()
    print("===================================================")
    print(c_model)
    print("===================================================")

    logger.info("Load prediction set ......")
    pred_dataset = Pred_Dataset(filename=VALID_SET, batch_size=batch_size)

    logger.info("Start predicate ......")
    pred(c_model, pred_dataset, VALID_SET)
