import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import BertForMaskedLM


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
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
        """
        super(CWTM, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased")

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

        self.fc = nn.Linear(in_features=768, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.latent_size)
        self.fc3 = nn.Linear(in_features=self.latent_size, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=768)
        self.fc5 = nn.Linear(in_features=768, out_features=1)

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
        for i, l in enumerate(self.encoder2):
            mu = l(local_embeddings, extended_attention_mask, None, output_attentions=True)[0]
        global_embeddings = mu[:, 0]

        kw = self.fc(local_embeddings)
        kw = F.leaky_relu(kw)
        kw = self.fc2(kw)
        kw = torch.softmax(kw, dim=2)

        for i, l in enumerate(self.encoder3):
            alphas = l(local_embeddings, extended_attention_mask, None, output_attentions=True)[0]
        alphas = torch.sigmoid(self.fc5(alphas))
        kw = kw * alphas
        kz = (kw * attention_masks.unsqueeze(2)).sum(1) + 1e-20
        theta = kz/kz.sum(1).unsqueeze(1) 

        reconstructed = F.leaky_relu(self.fc3(theta))
        reconstructed = self.fc4(reconstructed)

        hidden_embeddings = self.encoder.cls(local_embeddings)
        loss_fct = CrossEntropyLoss()
        mlm_loss = loss_fct(hidden_embeddings.view(-1, 30522), labels.view(-1))

        return local_embeddings, global_embeddings, mlm_loss, reconstructed, theta, kw