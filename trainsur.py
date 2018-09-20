import torch
import visdom
import logging
import numpy as np
import torch.nn as nn
from torch.nn import utils
import torch.optim as optim
from model.models import ClassificationModel
from utils.dataset import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# embed_mat = np.load('./mrc_data/vectors.npy')
# embed = nn.Embedding.from_pretrained(torch.Tensor(embed_mat))
# embed = nn.Embedding.from_pretrained(torch.randn([3,3]))
embed = nn.Embedding.from_pretrained(torch.randn([VOCAB.size(), 3]))
# embed = nn.Embedding(VOCAB.size(),3)
embed_dim = 3
hidden_dim = 5
lr = 0.001
batch_size = 32
weight_decay = 0.0001

vis = visdom.Visdom()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw(ep, acc, title):
    vis.line(X=torch.Tensor(1).fill_(ep).long(),
             Y=torch.Tensor(1).fill_(acc).float(),
             win=title,
             update='append' if ep != 0 else None,
             opts={'title': title, 'xlabel': 'EPOCH', 'ylabel': 'ACCURACY'})


def evaluate(model, validset):
    total = 0
    correct_total = 0
    model.eval()
    for ii in range(validset.__len__()):
        doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, aws = validset[ii]
        doc_pad = embed(doc_pad).to(device)
        qry_pad = embed(qry_pad).to(device)
        pred = model(doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask)
        pred = torch.argmax(pred, 1)
        correct_total += (pred == aws).sum().cpu().item()
        total += doc_pad.size(0)
        del doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, aws

    accuracy = correct_total / total
    return accuracy


def train(model: nn.Module, trainset: MRC_Dataset, epoch, validset: MRC_Dataset):
    loss_func = nn.CrossEntropyLoss()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    base_acc = 0.74
    for ep in range(epoch):

        model.train()

        trainset.shuffle()

        total = 0
        correct_total = 0

        for ii in range(trainset.__len__()):

            optimizer.zero_grad()

            doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, aws = trainset[ii]

            doc_pad = embed(doc_pad).to(device)
            qry_pad = embed(qry_pad).to(device)

            pred = model(doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask)

            loss = loss_func(pred, aws)

            total += doc_pad.size(0)
            pred = torch.argmax(pred, 1)
            correct = (pred == aws).sum().cpu().item()
            print(correct)
            correct_total += correct

            logger.info("ep: %d, loss: %f, correct: %d" % (ep, loss.item(), correct))

            if (ep * trainset.__len__() + ii + 1) % 10 == 0:
                vis.line(X=torch.Tensor(1).fill_(ep * trainset.__len__() + ii + 1).long(),
                         Y=torch.Tensor(1).fill_(loss.item()).float(),
                         win="Loss",
                         update='append' if ep != 0 or ii != 0 else None,
                         opts={'title': 'Loss', 'xlabel': 'Times', 'ylabel': 'Loss'})

            loss.backward()

            utils.clip_grad_value_(parameters, 5.0)

            optimizer.step()

            del doc_pad, doc_lens, doc_mask, qry_pad, qry_lens, qry_mask, loss

        train_accuracy = correct_total / total
        draw(ep, train_accuracy, "Train ACC")

        valid_accuracy = evaluate(model, validset)
        draw(ep, valid_accuracy, "Valid ACC")

        logger.info("ep: %d, train accuracy: %f, valid accuracy: %f" % (ep, train_accuracy, valid_accuracy))

        if valid_accuracy > base_acc:
            base_acc = valid_accuracy
            torch.save(model.state_dict(),
                       '%s_acc-%f_ep-%d.pth' % ('checkpoints/c_m', valid_accuracy, ep))


if __name__ == '__main__':
    logger.info("Start ......")

    logger.info("Create model ......")
    c_model = ClassificationModel(embed_dim, hidden_dim).to(device)
    print("===================================================")
    print(c_model)
    print("===================================================")

    logger.info("Load train set ......")
    train_set = MRC_Dataset(batch_size=batch_size, is_trainset=False)

    logger.info("Load valid set ......")
    # valid_set = MRC_Dataset(batch_size=batch_size, is_trainset=False)

    train(c_model, train_set, 50, train_set)






