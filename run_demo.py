import torch
from model import CWTM
from sklearn.datasets import fetch_20newsgroups

if __name__ == '__main__':

    corpus = fetch_20newsgroups(shuffle=True, 
                                subset="all",
                                random_state=1, 
                                categories=None, 
                                remove=('headers', 'footers', 'quotes'))
    print(corpus.data[:10])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [" ".join(s.split()[:64]) for s in corpus.data[:int(len(corpus.data)*0.7)]]
    model = CWTM(num_topics=20, backbone='bert-base-uncased', device=device)
    model.fit(texts, iterations=10, batch_size=16)

    model.get_topics(top_k=10)

    texts = corpus.data[:int(len(corpus.data)*0.3)]
    output = model.transform(texts)
    print(output['document_topic_distributions'].shape)
    print(output['word_topic_distributions'].shape)

    model.save(path="my save path")
    model.load(path="my save path")