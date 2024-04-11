## CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling

This is the source code for the LREC-COLING 2024 main conference paper:
[CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling](https://arxiv.org/abs/2305.09329v3).

## Training the model:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
texts = ['A woman is reading.', 'A man is playing a guitar.', 'A girl is eating an apple.', 'A boy is sitting under a tree.']
model = CWTM(num_topics=20, iterations=20, backbone='bert-base-uncased', device=device)        
model.fit(texts)
```
## Transform texts according to the fitted model

```python
texts = ['A woman is reading.', 'A man is playing a guitar.', 'A girl is eating an apple.', 'A boy is sitting under a tree.']
theta = model.transform(texts)
```
