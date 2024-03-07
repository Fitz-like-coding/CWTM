import os
import re
import copy
import time
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertTokenizerFast, AdamW, get_scheduler
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
import nltk
import json
from nltk.corpus import stopwords
nltk.download('stopwords')
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.prefix_projection = True
        self.pre_seq_len = 10
        self.prefix_hidden_size = 512
        self.num_hidden_layers = 12
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.prefix_hidden_size, self.num_hidden_layers * 2 * self.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(self.pre_seq_len, self.num_hidden_layers * 2 * self.hidden_size)

    # def forward(self, prefix_tokens: torch.Tensor):
    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

class CWTM(nn.Module):
    def __init__(self, latent_size = 20, device = "cpu"):
        """
        Initialize InferenceNetwork.
        Args
            latent_size : int, number of topic components, (default 20)
        """
        super(CWTM, self).__init__()
        self.device = device
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.latent_size = latent_size
        self.hidden_size = self.encoder.config.hidden_size
        self.vocab_size = self.encoder.config.vocab_size
        self.word2topic = {}
        self.docCountByWord = {}

        self.prefix_encoder = PrefixEncoder().to(self.device)
        self.pre_seq_len = 10
        self.n_layer = 12
        self.n_head = 12 
        self.n_embd = 64
        self.prefix_tokens = torch.arange(self.pre_seq_len).long().to(self.device)
        for param in self.prefix_encoder.parameters():
            param.requires_grad = True

        for param in self.encoder.bert.parameters():
            param.requires_grad = False

        for param in self.encoder.cls.parameters():
            param.requires_grad = False

        self.encoder2 = copy.deepcopy(self.encoder.bert.encoder.layer[-1:])
        for param in self.encoder2.parameters():
            param.requires_grad = True

        self.encoder3 = copy.deepcopy(self.encoder.bert.encoder.layer[-1:])
        for param in self.encoder3.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(in_features=self.hidden_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.latent_size)
        self.fc3 = nn.Linear(in_features=self.latent_size, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=768)
        self.fc5 = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.dropout = nn.Dropout(p=0.5)

        [self._init_weights(p) for n, p in self.encoder2.named_parameters()]
        [self._init_weights(p) for n, p in self.encoder3.named_parameters()]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
        
    def forward(self, input_ids, attention_masks, labels=None):
        """Forward pass."""
        past_key_values = self.get_prompt(input_ids.size(0))
        prefix_attention_mask = torch.ones(input_ids.shape[0], self.pre_seq_len).to(self.device)
        attention_masks_prom = torch.cat((prefix_attention_mask, attention_masks), dim=1)
        
        # get word embedding from Bert
        features = self.encoder.bert(input_ids, 
                                attention_masks_prom, 
                                head_mask=None, 
                                output_hidden_states=True, 
                                output_attentions=True,
                                past_key_values=past_key_values[:12])
        
        local_embeddings = features.hidden_states[-1]

        extended_attention_mask = self.encoder.get_extended_attention_mask(attention_masks, input_ids.size(), self.device)
        for _, l in enumerate(self.encoder2):
            mu = l(local_embeddings, extended_attention_mask, None, output_attentions=True)[0]
        global_embeddings = mu[:, 0]

        kw = self.fc(local_embeddings)
        kw = F.leaky_relu(kw)
        kw = self.fc2(kw)
        kw = torch.softmax(kw, dim=2)

        for _, l in enumerate(self.encoder3):
            alphas = l(local_embeddings, extended_attention_mask, None, output_attentions=True)[0]
        alphas = torch.sigmoid(self.fc5(alphas))
        word_topic = kw * alphas
        kz = (word_topic * attention_masks.unsqueeze(2)).sum(1) + 1e-20
        doc_topic = kz/kz.sum(1).unsqueeze(1) 

        reconstructed = F.leaky_relu(self.fc3(doc_topic))
        reconstructed = self.fc4(reconstructed)

        hidden_embeddings = self.encoder.cls(local_embeddings)
        loss_fct = CrossEntropyLoss()
        mlm_loss = loss_fct(hidden_embeddings.view(-1, self.vocab_size), labels.view(-1))

        return local_embeddings, global_embeddings, mlm_loss, reconstructed, doc_topic, word_topic
    
    def getMaskedInput(self, input_ids, words_mask, rate = 0.15, mask_rate = 0.8, replace_rate = 0.1):
        p = torch.ones(input_ids.size()) * rate
        noise_mask = torch.bernoulli(p).to(self.device)
        fake_index = (noise_mask * words_mask) == 1
        input = input_ids.clone()

        p = torch.ones(input_ids.size()) * mask_rate
        noise_mask2 = torch.bernoulli(p).to(self.device)
        fake_index2 = (noise_mask2 * (noise_mask * words_mask)) == 1
        input[fake_index2] = 103

        p = torch.ones(input_ids.size()) * replace_rate
        noise_mask3 = torch.bernoulli(p).to(self.device)
        fake_index3 = (noise_mask3 * (noise_mask * words_mask - noise_mask * words_mask * noise_mask2)) == 1
        input[fake_index3] = torch.randint(0, self.vocab_size, (fake_index3.sum(),)).to(self.device)
        return input
    
    def create_masks(self, lens_a):
        pos_mask = torch.zeros((np.sum(lens_a), len(lens_a))).to(self.device)
        neg_mask = torch.ones((np.sum(lens_a), len(lens_a))).to(self.device)
        temp = 0
        for idx in range(len(lens_a)):
            for j in range(temp, lens_a[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += lens_a[idx]
        return pos_mask, neg_mask
        
    def get_optimizer(self, lr=1e-3, eps=1e-6, weight_decay=0.01):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer = AdamW([
                {'params': self.fc.parameters()},
                {'params': self.fc2.parameters()},
                {'params': self.fc3.parameters()},
                {'params': self.fc4.parameters()},
                {'params': self.fc5.parameters()},
                {'params': self.prefix_encoder.parameters()},
                {'params': [p for n, p in self.encoder2.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in self.encoder2.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in self.encoder3.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in self.encoder3.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ],
            lr=lr, eps=eps, weight_decay=weight_decay, correct_bias=True)
        return optimizer
    
    def get_scheduler(self, n_epochs, optimizer, data_generator):
        num_training_steps = n_epochs * len(data_generator)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_training_steps * 0.1,
            num_training_steps=num_training_steps * 0.9
        )
        return lr_scheduler

    def load(self, path):
        model_state_dict = torch.load(path+"/model_state_dict.pth")
        self.load_state_dict(model_state_dict)
        with open(path+'/word2topic.txt', "r") as file:
            for line in tqdm(file.readlines()):
                temp = json.loads(line)
                self.word2topic[temp['word']] = temp['topic_dist']
        print(f"Load model from {path}.")
    
    def save(self, path=None):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.state_dict(), f"{path}/model_state_dict.pth")
        with open(f"{path}/word2topic.txt", "w") as file:
            for w, topic_dist in self.word2topic.items():
                file.write(json.dumps({"word": str(w), "topic_dist": [round(float(v), 5) for v in topic_dist]}))
                file.write("\n")
        print(f"Save model to {path}.")

    def fit(self, data_generator, n_epochs):
        optimizer = self.get_optimizer(lr=1e-3, eps=1e-6, weight_decay=0.01)
        lr_scheduler = self.get_scheduler(n_epochs, optimizer, data_generator)
        for epoch in range(n_epochs):
            start_time = time.time()

            self.train()
            train_loss = []
            train_topic_word_loss = []
            train_document_topic_loss = []
            train_mutual_information_loss = []
            train_mlm_loss = []
            train_recon_loss = []
            for batch_idx, sample in tqdm(enumerate(data_generator), total=len(data_generator)):
                input = sample[0].to(self.device)
                attention_masks = sample[1].to(self.device)
                words_mask = sample[2].to(self.device)
                cls = sample[3].to(self.device)

                optimizer.zero_grad()
                masked_input = self.getMaskedInput(input, words_mask)
                labels = input.clone()
                labels[labels==0] = -100
                local_embeddings, global_embeddings, mlm_loss, reconstructed, document_topic, topic_word = self(masked_input, attention_masks, labels)
                topic_word_loss = self.topic_word_loss(topic_word, words_mask, input)
                document_topic_loss = self.document_topic_loss(document_topic)
                mutual_information_loss = self.mutual_information_loss(local_embeddings, global_embeddings, attention_masks)
                recon_loss = nn.MSELoss()(global_embeddings, reconstructed)
                loss = topic_word_loss + document_topic_loss + 1.0 * mutual_information_loss + 1.0 * mlm_loss + 1.0 * recon_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                train_loss.append(loss.item())
                train_topic_word_loss.append(topic_word_loss.item())
                train_document_topic_loss.append(document_topic_loss.item())
                train_mutual_information_loss.append(mutual_information_loss.item())
                train_mlm_loss.append(mlm_loss.item())
                train_recon_loss.append(recon_loss.item())
                torch.cuda.empty_cache()

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
            print(f'\ttrain Loss: {np.mean(train_loss):.4f}')
            print(f'\ttrain train_topic_word_loss: {np.mean(train_topic_word_loss):.4f}')
            print(f'\ttrain train_document_topic_loss: {np.mean(train_document_topic_loss):.4f}')
            print(f'\ttrain train_mutual_information_loss: {np.mean(train_mutual_information_loss):.4f}')
            print(f'\ttrain train_mlm_loss: {np.mean(train_mlm_loss):.4f}')
            print(f'\ttrain train_recon_loss: {np.mean(train_recon_loss):.4f}')

        print("extracting topic words...")
        stopwords = set()
        with open("./data/stopwords.en.txt") as file:
            for word in file.readlines():
                stopwords.add(word.strip())
        stopwords2 = []
        for word in stopwords:
            stopwords2.extend(self.tokenizer.tokenize(word))
        stopwords.update(stopwords2)
        self.extracting_topics(data_generator, stopwords=stopwords)
        current_time = time.time()
        self.save(path=f"./save/CWTM_{self.latent_size}_topics_{current_time}")
        print("Training finished")
        return 
    
    def extracting_topics(self, data, min_df=3, max_df=0.5, remove_top=10, stopwords=[]):
        lemmatizer = WordNetLemmatizer()
        self.eval()
        doc_count = 0
        for i, sample in tqdm(enumerate(data), total=len(data)):
            input_ids = sample[0].to(self.device)
            attention_masks = sample[1].to(self.device)
            words_mask = sample[2].to(self.device)
            cls = sample[3].to(self.device)
            sents = sample[4]
            with torch.no_grad():
                _, _, _, _, theta, kw = self(input_ids, attention_masks.clone(), input_ids)
                kw = kw * words_mask.unsqueeze(2)
                for sent_idx, input_id in enumerate(input_ids):
                    doc_count += 1
                    previous_token = "[CLS]"
                    previous_topic = kw[sent_idx][0].cpu().data.numpy()
                    c = 1
                    temp_doc_count = {}
                    for word_idx, current_token in enumerate(self.tokenizer.convert_ids_to_tokens(input_id)):
                        if word_idx == 0:
                            continue
                        if current_token.startswith("##"):
                            previous_token += current_token.replace("##", "")
                            previous_topic += kw[sent_idx][word_idx].cpu().data.numpy()
                            c += 1
                        else:
                            previous_topic = previous_topic/c
                            if previous_token not in ["[SEP]", "[PAD]", "[CLS]"] and not re.search("\W|\d", previous_token) and previous_token not in stopwords:
                                token = lemmatizer.lemmatize(previous_token)
                                temp_doc_count[token] = 1
                                dist = previous_topic 
                                try:
                                    self.word2topic[token] += dist
                                except:
                                    self.word2topic[token] = np.zeros(self.latent_size)
                                    self.word2topic[token] += dist
                            previous_token = current_token
                            previous_topic = kw[sent_idx][word_idx].cpu().data.numpy()
                            c = 1
                    for w in temp_doc_count:
                        self.docCountByWord[w] = self.docCountByWord.get(w, 0) + temp_doc_count[w]
            torch.cuda.empty_cache()

        for w, c in sorted(self.docCountByWord.items(), key=lambda x:x[1], reverse=True)[:remove_top]:
            self.word2topic.pop(w, None)
        for w in self.docCountByWord:
            if type(min_df) == int and self.docCountByWord[w] < min_df:
                self.word2topic.pop(w, None)
            elif type(min_df) == float and self.docCountByWord[w]/len(data.dataset) < min_df:
                self.word2topic.pop(w, None)

            if type(max_df) == int and self.docCountByWord[w] > max_df:
                self.word2topic.pop(w, None)
            elif type(max_df) == float and self.docCountByWord[w]/len(data.dataset) > max_df:
                self.word2topic.pop(w, None) 


    def transform(self, data):
        self.eval()       
        X = []
        for _, sample in enumerate(data):
                input_ids = sample[0].to(self.device)
                attention_masks = sample[1].to(self.device)
                words_mask = sample[2].to(self.device)
                with torch.no_grad():
                    # for topic model
                    _, _, _, _, theta, _ = self(input_ids.clone(), attention_masks.clone(), input_ids.clone())
                    X.extend(theta.cpu().data.numpy())
        X = np.array(X)
        return X
    
    def get_topics(self, top_k = 10):
        vobs = np.array(list(self.word2topic.keys()))
        topic2word = np.array(list(self.word2topic.values())).T
        topics = {}
        for i in range(self.latent_size):
            topics[i] = [w for w in vobs[topic2word[i].argsort()[::-1]]][:top_k]
            topic = ", ".join(topics[i])
            print('topic {index}: {words}'.format(index = i, words=topic))
        return topics
    
    def mmd_loss(self, x, y, t=1.0, kernel='diffusion'):
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

        off_diag = 1 - torch.eye(n, device=self.device)
        k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
        k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
        k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
        sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
        sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
        sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy

    def topic_word_loss(self, word_topic, words_mask, input):
        word_topic_squeezed = word_topic.reshape((-1, word_topic.size(2)))
        word_topic_squeezed = word_topic_squeezed[words_mask.reshape(-1) == 1, :]

        word_ids = input.flatten()[words_mask.reshape(-1) == 1]
        masks = word_ids.view(word_ids.size(0), 1).expand(-1, word_topic_squeezed.size(1))
        res = torch.zeros((self.vocab_size, self.latent_size), dtype=torch.float, device=self.device).scatter_add_(0, masks, word_topic_squeezed)
        topic_word = res.t()/res.t().sum(1).unsqueeze(1)

        diric_prior = np.random.dirichlet(np.ones(topic_word.size(1)) * 0.1, size=topic_word.size(0))
        diric_prior = torch.tensor(diric_prior, device=self.device, dtype=torch.float32)
        topic_word_loss = self.mmd_loss(diric_prior, topic_word)
        return topic_word_loss
    
    def document_topic_loss(self, document_topic):
        diric_prior = np.random.dirichlet(np.ones(document_topic.size(1)) * 0.1, size=document_topic.size(0))
        diric_prior = torch.tensor(diric_prior, device=self.device, dtype=torch.float32)
        doc_topic_loss = self.mmd_loss(diric_prior, document_topic)
        return doc_topic_loss
    
    def mutual_information_loss(self, local_embeddings, global_embeddings, attention_masks):
        sentence_lengths = torch.clamp(attention_masks.sum(1), min=0).data.cpu().numpy().astype(int)
        pos_mask, neg_mask = self.create_masks(sentence_lengths)

        local_embeddings_squeezed = local_embeddings.reshape((-1, local_embeddings.size(2)))
        local_embeddings_squeezed = local_embeddings_squeezed[attention_masks.reshape(-1) == 1, :]
        p_samples = torch.mm(local_embeddings_squeezed, global_embeddings.t())
        p_samples = torch.sigmoid(p_samples/np.sqrt(global_embeddings.size(1)))

        pos_samples = p_samples * pos_mask
        neg_samples = p_samples * neg_mask
        E_pos = -torch.log(pos_samples + 1e-6) * pos_mask
        E_pos = E_pos.sum()/pos_mask.sum()
        E_neg = -torch.log(1 - neg_samples + 1e-6) * neg_mask
        E_neg = E_neg.sum()/neg_mask.sum()
        return E_pos + E_neg