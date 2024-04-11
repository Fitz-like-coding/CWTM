## CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling

This is the source code for the LREC-COLING 2024 main conference paper:
[CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling](https://arxiv.org/abs/2305.09329v3).

**Training the model**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
texts = ['A woman is reading.', 'A man is playing a guitar.', 'A girl is eating an apple.', 'A boy is sitting under a tree.']
model = CWTM(num_topics=20, backbone='bert-base-uncased', device=device)        
model.fit(texts, iterations=20)
```

**Print top words of each topic**
```python
model.get_topics(top_k=10)
```

**Transform texts to get document-topic and word-topic distributions**
```python
texts = ['A woman is reading.', 'A man is playing a guitar.', 'A girl is eating an apple.', 'A boy is sitting under a tree.']
output = model.transform(texts)
print(output['document_topic_distributions'])
print(output['word_topic_distributions'])
```

**save fitted model**
```python
model.save(path="my save path")
```

**load fitted model**
```python
model.load(path="my save path")
```
