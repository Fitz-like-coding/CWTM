## CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling

This is the source code for the LREC-COLING 2024 main conference paper:[CWTM: Leveraging Contextualized Word Embeddings from BERT for Neural Topic Modeling](https://arxiv.org/abs/2305.09329v3).

# Training the model:

* run run.py to reproduce the coherence, diversity and classification results in the paper.

# use your own dataset:
   
   1. create a dataset following the datasets template in utils/load_datasets.py

```python
sub_train = TwentyNewsDataset(subset="train")
train_data = DataLoader(sub_train, batch_size=16, shuffle=True, num_workers = 4, pin_memory=True, collate_fn=generate_batch)
```

   2. train the model

```python
N_EPOCHS = 20
cwtm_model = CWTM(latent_size = 20, device = device).to(device)        
cwtm_model.fit(train_data, N_EPOCHS)
```

   3. transform texts

```python
sub_valid = TwentyNewsDataset(subset="valid")
valid_data = DataLoader(sub_valid, batch_size=16, shuffle=False, num_workers = 4, pin_memory=True, collate_fn=generate_batch)
X = cwtm_model.transform(valid_data)
```
