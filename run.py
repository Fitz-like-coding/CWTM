
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
from transformers import BertTokenizerFast
import torch
from torch import nn
from transformers import AdamW
from transformers import get_scheduler  
import nltk
from load_datasets import TwentyNewsDataset, TagMyNewsDataset, Dbpedia14Dataset, TwitterEmotion3, AGNews
from sklearn.metrics.cluster import contingency_matrix
from model import CWTM
import re

import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('wordnet')

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["TOKENIZERS_PARALLELISM"]="false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
def generate_batch(batch):
    encoding = tokenizer([str(entry["texts"]) for entry in batch], return_tensors="pt", padding="longest", truncation=True, max_length=502)
    label = torch.tensor([entry["labels"][0] for entry in batch])

    words_mask = encoding['attention_mask'].clone()
    length = encoding['attention_mask'].sum(1)

    words_mask[torch.arange(length.size(0)), 0] = 0
    words_mask[torch.arange(length.size(0)), length-1] = 0

    sentences = [str(entry["texts"]) for entry in batch]
    return encoding['input_ids'].type(torch.LongTensor), encoding['attention_mask'].type(torch.LongTensor), words_mask.type(torch.LongTensor), label.type(torch.LongTensor), sentences


def mmd_loss(x, y, t=1.0, kernel='diffusion'):
    '''
    computes the mmd loss with information diffusion kernel
    :param x: dirichlet prior
    :param y: encoded x from encoder network
    :param t:
    :return:
    '''
    eps = 1e-6
    n, d = x.shape
    qx = torch.sqrt(torch.clamp(x, eps, 1))
    qy = torch.sqrt(torch.clamp(y, eps, 1))
    xx = torch.mm(qx, qx.T)
    yy = torch.mm(qy, qy.T)
    xy = torch.mm(qx, qy.T)

    def diffusion_kernel(a, tmpt, dim):
        return torch.exp(-torch.square(torch.acos(a)) / tmpt)

    off_diag = 1 - torch.eye(n, device=device)
    k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
    k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
    k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
    sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
    sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
    sum_xy = 2 * k_xy.sum() / (n * n)
    return sum_xx + sum_yy - sum_xy

def create_masks(lens_a):
      pos_mask = torch.zeros((np.sum(lens_a), len(lens_a))).to(device)
      neg_mask = torch.ones((np.sum(lens_a), len(lens_a))).to(device)
      temp = 0
      for idx in range(len(lens_a)):
          for j in range(temp, lens_a[idx] + temp):
              pos_mask[j][idx] = 1.
              neg_mask[j][idx] = 0.
          temp += lens_a[idx]

      return pos_mask, neg_mask

def purity_score(y_true, y_pred):
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def cluster_acc(Y_pred, Y):
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def read_stopwords(fn):
    return set([line.strip() for line in open(fn, encoding='utf-8') if len(line.strip()) != 0])

if __name__ == "__main__":
    # for config_dataset in ["20news", "TagMyNews", "Dbpedia14", "TwitterEmotion", "AGNews"]:
    config_dataset = "20news"
    if config_dataset == "20news":
        N_EPOCHS = 1
        sub_train = TwentyNewsDataset(subset="train")
        sub_valid = TwentyNewsDataset(subset="test")
    elif config_dataset == "TagMyNews":
        N_EPOCHS = 20
        sub_train = TagMyNewsDataset(subset="train")
        sub_valid = TagMyNewsDataset(subset="test")
    elif config_dataset == "Dbpedia14":
        N_EPOCHS = 1
        sub_train = Dbpedia14Dataset(subset="train")
        sub_valid = Dbpedia14Dataset(subset="test")
    elif config_dataset == "TwitterEmotion":
        N_EPOCHS = 20
        sub_train = TwitterEmotion3(subset="train")
        sub_valid = TwitterEmotion3(subset="test")
    elif config_dataset == "AGNews":
        N_EPOCHS = 1
        sub_train = AGNews(subset="train")
        sub_valid = AGNews(subset="test")


    BATCH = 16
    latent_size = 20
    cwtm_model = CWTM(latent_size = latent_size, device = device).to(device)        

    train_data = DataLoader(sub_train, batch_size=BATCH, shuffle=True, num_workers = 4, pin_memory=True, collate_fn=generate_batch)
    cwtm_model.fit(train_data, N_EPOCHS)

    valid_data = DataLoader(sub_valid, batch_size=BATCH, shuffle=False, num_workers = 4, pin_memory=True, collate_fn=generate_batch)
    X = cwtm_model.transform(valid_data)
    Y = sub_valid['labels'][:]

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans

    clf = LogisticRegression(max_iter=500)
    valid_cls_scores = cross_val_score(clf, X, Y, scoring="accuracy", cv=5, n_jobs=5)
    print("valid_cls_scores: " + str(np.mean(valid_cls_scores)))
