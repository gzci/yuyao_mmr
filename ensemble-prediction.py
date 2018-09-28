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

embed_mat = np.load('./mrc_data/vectors.npy')
embed = nn.Embedding.from_pretrained(torch.Tensor(embed_mat))

# hyperparameter
embed_dim = 200
hidden_dim = 128
batch_size = 64

VALID_SET = "./mrc_data/ai_challenger_oqmrc_validationset.json"
TESTA_SET = "./mrc_data/ai_challenger_oqmrc_testa.json"

model0 = "./checkpoints/model128_0.751.pth"
model1 = "./checkpoints/model128_0.753.pth"
model2 = "./checkpoints/model128_0.748.pth"
model3 = "./checkpoints/model88_0.752.pth"
model4 = "./checkpoints/model88_0.748.pth"
#model5 = "./checkpoints/model192_0.749.pth"
models = [model0, model1, model2, model3, model4]
models_dim = [128, 128, 128, 88, 88, 192]


def write_to_file(pred_res, filename):
    pred_res = sorted(pred_res.items(), key=lambda item: item[0])
    with open(filename, "w") as pred_file:
        for res in pred_res:
            idx, answer = res
            pred_file.write("%d\t%s\n" % (idx, answer))


def random_guess(pred_res, filename, name=TESTA_SET):
    logger.info(str(len(invalid_items)))
    for it in invalid_items:
        print(str(it["alternatives"]))
        pred_res[it["query_id"]] = str(it["alternatives"]).strip().split("|")[0]
    logger.info(str(len(pred_res)))
    write_to_file(pred_res, filename)


def transform(pred_, alternatives):
    res = []
    for i in range(len(pred_)):
        p_label = pred_[i]
        al = alternatives[i]
        res.append(al[p_label].strip())
    return res


def pred(model_s, dataset, name=TESTA_SET):

    pred_res = {}

    for ii in range(dataset.__len__()):
        doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, als, qry_ids, w_ms, c_ms, q_w_ms, q_c_ms, alss = dataset[ii]
        pred_sum = torch.zeros([doc_pad.size(0), 3]).to(device)
        for model in model_s:
            pred_ = model(doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, w_ms, c_ms, q_w_ms, q_c_ms, als)
            pred_sum.add_(pred_)
        pred_ = torch.argmax(pred_sum, 1).cpu().tolist()
        pred_ = transform(pred_, alss)
        for idx, answer in zip(qry_ids, pred_):
            pred_res[idx] = answer
            logger.info("%d:%s" % (idx, answer))

    if (len(pred_res) == 10000 and name == TESTA_SET) or (len(pred_res) == 30000 and name == VALID_SET):
        write_to_file(pred_res, "ensemble_prediction.txt")
    else:
        logger.info(str(len(pred_res)))
        random_guess(pred_res, "ensemble_prediction.txt", name=name)


if __name__ == '__main__':

    logger.info("Start ......")

    logger.info("Load model ......")

    mds = []
    for ii in range(len(models)):
        c_model = ClassificationModel(embed_dim, models_dim[ii])
        c_model.load_state_dict(torch.load(models[ii]))
        c_model.to(device)
        c_model.eval()
        mds.append(c_model)

    print("===================================================")
    print(c_model)
    print("===================================================")

    logger.info("Load prediction set ......")
    pred_dataset = Pred_Dataset(filename=TESTA_SET, batch_size=batch_size)

    logger.info("Start predicate ......")
    pred(mds, pred_dataset, TESTA_SET)
