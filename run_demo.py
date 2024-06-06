import torch
from model import CWTM
from sklearn.datasets import fetch_20newsgroups

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = ['A woman is reading.', 'A man is playing a guitar.', 'A girl is eating an apple.', 'A boy is sitting under a tree.']
    
    # backbone must be a bert based model
    model = CWTM(num_topics=2, backbone='bert-base-uncased', device=device)
    model.fit(texts, iterations=10, batch_size=16)

    stopwords = set()
    with open("./data/stopwords.en.txt") as file:
        for word in file.readlines():
            stopwords.add(word.strip())
    model.extracting_topics(texts, min_df=1, max_df=1.0, remove_top=0, stopwords=stopwords)
    model.get_topics(top_k = 10)

    texts = ['A woman is reading.', 'A man is playing a guitar.', 'A girl is eating an apple.', 'A boy is sitting under a tree.']
    output = model.transform(texts)
    print(output['document_topic_distributions'].shape)
    print(output['word_topic_distributions'].shape)

    model.save(path="my save path")
    model.load(path="my save path")