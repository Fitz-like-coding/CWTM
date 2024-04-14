
from torch.utils.data import DataLoader
import numpy as np
import torch
from transformers import BertTokenizerFast
import torch
import nltk
from utils.load_datasets import TwentyNewsDataset, TagMyNewsDataset, Dbpedia14Dataset, TwitterEmotion3, AGNews
from utils.evaluation import CoherenceEvaluator, DiversityEvaluator
from sklearn.metrics.cluster import contingency_matrix
from model import CWTM
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('wordnet')

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["TOKENIZERS_PARALLELISM"]="false"

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print (device)

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

if __name__ == "__main__":
    # for config_dataset in ["20news", "TagMyNews", "Dbpedia14", "TwitterEmotion", "AGNews"]:
    config_dataset = "TwitterEmotion"
    if config_dataset == "20news":
        N_EPOCHS = 20
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
    cwtm_model.fit(train_data, N_EPOCHS)

    # cwtm_model.load("./save/CWTM_20_topics_1709658726.626927")
    topics = cwtm_model.get_topics(10)

    print("Extracting Coherence score...")
    evaluator = CoherenceEvaluator()
    score = evaluator.coherence_score(topics.values())
    print("Coherence score:", score)

    print("Extracting diversity score...")
    evaluator = DiversityEvaluator(device=device, target_model=cwtm_model)
    evaluator.fit_embeddings(train_data)
    print("Diversity score:", evaluator.diversity_score(topics))

    valid_data = DataLoader(sub_valid, batch_size=BATCH, shuffle=False, num_workers = 1, pin_memory=False, collate_fn=generate_batch)
    X = cwtm_model.transform(valid_data)
    Y = sub_valid[:]['labels']

    clf = LogisticRegression(max_iter=500)
    valid_cls_scores = cross_val_score(clf, X, Y, scoring="accuracy", cv=5, n_jobs=5)
    print("Classification score: " + str(np.mean(valid_cls_scores)))
