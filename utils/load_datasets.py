from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset
from preprocessor import *
import pandas as pd
import numpy as np
import torch
import re
stopwords = read_stopwords("../data/stopwords.en.txt")
processor = preprocessor(stopwords)

class TwentyNewsDataset(Dataset):
    """20news dataset."""

    def __init__(self, subset="train"):
        """
        Args:
            subset (string): Load train or test dataset.
        """
        assert subset in ["train", "test"], "subset should be set either to train or test."

        corpus = fetch_20newsgroups(shuffle=True, 
                                    subset="all",
                                    random_state=1, 
                                    categories=None, 
                                    remove=('headers', 'footers', 'quotes'))
        texts = []
        labels = []
        for i in range(len(corpus.data[:])):
            # add this block of code if want to reprode the results in the paper
            temp = " ".join([w for w in corpus.data[i].split()])
            temp2 = processor.preprocess(temp)
            if len(temp2) == 0:
                continue
            #

            texts.append(" ".join([w for w in corpus.data[i].split()]))
            labels.append(corpus.target[i])

        if subset == "train":
            texts = texts[:int(len(labels)*0.7)]
            labels = labels[:int(len(labels)*0.7)]
        if subset == "test":
            texts = texts[int(len(labels)*0.7):]
            labels = labels[int(len(labels)*0.7):]
        self.corpus = pd.DataFrame(list(zip(texts, labels)), columns =['texts', 'labels'])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts =  np.array(self.corpus.iloc[idx]["texts"])
        labels =  np.array(self.corpus.iloc[idx]["labels"])
        labels = labels.ravel()
        sample = {'texts': texts,
                  'labels': labels}
        return sample
    
# 7 categorics 
class TagMyNewsDataset(Dataset):
    """TagMyNews dataset."""

    def __init__(self, subset="train"):
        """
        Args:
            subset (string): Load train or test dataset.
        """
        assert subset in ["train", "test"], "subset should be set either to train or test."

        data = []
        targets = []
        with open("./data/TagMyNews.txt", "r") as file:
            c = 0
            temp = ''
            for line in file.readlines():
                if c % 8 == 0 or c % 8 == 1:
                    temp += line.strip() + ' '
                    c += 1
                    continue
                elif c % 8 == 2:
                    temp = re.sub("u\.s\.", "usa", temp)
                    temp = re.sub("\\bu\.s\\b", "usa", temp)
                    temp = re.sub("\\bus\\b", "usa", temp)
                    data.append(temp.strip())
                    temp = ''
                    c += 1
                elif c % 8 == 6:
                    targets.append(line.strip())
                    c += 1
                else:
                    c += 1
                    continue

        texts = []
        labels = []
        for i in range(len(data[:])):
            temp = " ".join([w for w in data[i].split()])
            temp2 = processor.preprocess(temp)
            if len(temp2) == 0:
                continue
            texts.append(" ".join([w for w in data[i].split()]))
            labels.append(["business", "entertainment", "health", "sci_tech", "sport", "us", "world"].index(targets[i]))
        
        if subset == "train":
            texts = texts[:int(len(labels)*0.7)]
            labels = labels[:int(len(labels)*0.7)]
        if subset == "test":
            texts = texts[int(len(labels)*0.7):]
            labels = labels[int(len(labels)*0.7):]

        self.corpus = pd.DataFrame(list(zip(texts, labels)), columns =['texts',  'labels'])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts =  np.array(self.corpus.iloc[idx]["texts"])
        labels =  np.array(self.corpus.iloc[idx]["labels"])
        labels = labels.ravel()
        sample = {'texts': texts,
                  'labels': labels}
        return sample
    
class Dbpedia14Dataset(Dataset):
    """dbpedia14 dataset"""

    def __init__(self, subset="train"):
        """
        Args:
            subset (string): Load train or test dataset.
        """
        assert subset in ["train", "test"], "subset should be set either to train or test."
        data = load_dataset("dbpedia_14", split=subset)

        texts = []
        labels = []
        for i in range(len(data)):
            temp = " ".join([w for w in data[i]['content'].split()])
            temp2 = processor.preprocess(temp)
            if len(temp2) == 0:
                continue
            texts.append(" ".join([w for w in data[i]['content'].split()]))
            labels.append(data[i]["label"])

        self.corpus = pd.DataFrame(list(zip(texts, labels)), columns =['texts', 'labels'])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts =  np.array(self.corpus.iloc[idx]["texts"])
        labels =  np.array(self.corpus.iloc[idx]["labels"])
        labels = labels.ravel()
        sample = {'texts': texts,
                  'labels': labels}
        return sample

class TwitterEmotion3(Dataset):
    """20news dataset."""

    def __init__(self, subset="train"):
        """
        Args:
            subset (string): Load train or test dataset.
        """
        assert subset in ["train", "test"], "subset should be set either to train or test."
        data = load_dataset("tweet_eval", "emotion", split=subset)

        texts = []
        labels = []
        for i in range(len(data)):
            temp = " ".join([w for w in data[i]['text'].split()])
            temp2 = processor.preprocess(temp)
            if len(temp2) == 0:
                continue
            texts.append(" ".join([w for w in data[i]['text'].split()]))
            labels.append(data[i]["label"])

        self.corpus = pd.DataFrame(list(zip(texts, labels)), columns =['texts', 'labels'])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts = np.array(self.corpus.iloc[idx]["texts"])
        labels = np.array(self.corpus.iloc[idx]["labels"])
        labels = labels.ravel()
        sample = {'texts': texts,
                  'labels': labels}
        return sample

class AGNews(Dataset):
    """20news dataset."""

    def __init__(self, subset="train"):
        """
        Args:
            subset (string): Load train or test dataset.
        """
        assert subset in ["train", "test"], "subset should be set either to train or test."
        data = load_dataset("ag_news", split=subset)

        texts = []
        labels = []
        for i in range(len(data)):
            temp = " ".join([w for w in data[i]['text'].split()])
            temp2 = processor.preprocess(temp)
            if len(temp2) == 0:
                continue
            texts.append(" ".join([w for w in data[i]['text'].split()]))
            labels.append(data[i]["label"])

        self.corpus = pd.DataFrame(list(zip(texts, labels)), columns =['texts', 'labels'])

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        texts = np.array(self.corpus.iloc[idx]["texts"])
        labels = np.array(self.corpus.iloc[idx]["labels"])
        labels = labels.ravel()
        sample = {'texts': texts,
                  'labels': labels}
        return sample
    
if __name__ == "__main__":
   for config_dataset in ["AGNews"]:
        if config_dataset == "20news":
            sub_train = TwentyNewsDataset(subset="train")
            sub_valid = TwentyNewsDataset(subset="test")
        elif config_dataset == "TagMyNews":
            sub_train = TagMyNewsDataset(subset="train")
            sub_valid = TagMyNewsDataset(subset="test")
        elif config_dataset == "Dbpedia14":
            sub_train = Dbpedia14Dataset(subset="train")
            sub_valid = Dbpedia14Dataset(subset="test")
        elif config_dataset == "TwitterEmotion":
            sub_train = TwitterEmotion3(subset="train")
            sub_valid = TwitterEmotion3(subset="test")
        elif config_dataset == "AGNews":
            sub_train = AGNews(subset="train")
            sub_valid = AGNews(subset="test")

        print(len(sub_train), len(sub_valid))