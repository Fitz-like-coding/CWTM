import numpy as np
from palmettopy.palmetto import Palmetto
from transformers import AutoModel, BertTokenizerFast
from nltk.stem import WordNetLemmatizer
import torch
from itertools import combinations
from scipy.spatial.distance import cosine

def centroid_score(model_output, topk):
    """
    Retrieves the score of the metric

    :return WECS
    """
    topics = model_output['topics']
    sim = 0
    count = 0
    for list1, list2 in combinations(topics, 2):
        centroid1 = np.zeros(768)
        centroid2 = np.zeros(768)
        count1, count2 = 0, 0
        for word1 in list1[:topk]:
            centroid1 = centroid1 + word1[1]
            count1 += 1
        for word2 in list2[:topk]:
            centroid2 = centroid2 + word2[1]
            count2 += 1
        centroid1 = centroid1 / count1
        centroid2 = centroid2 / count2
        sim = sim + cosine(centroid1, centroid2)
        count += 1
    return sim / count

class CoherenceEvaluator:
    def __init__(self, endpoint="http://palmetto.aksw.org/palmetto-webapp/service/"):
        self.palmetto = Palmetto(endpoint, timeout=30)

    def get_coherence(self, words, coherence_type="cv"):
        return self.palmetto.get_coherence(words, coherence_type=coherence_type)
    
    def coherence_score(self, topics, coherence_type="cv"):
        scores = []
        for words in topics:
            score = self.get_coherence(words, coherence_type)
            scores.append(score)
            print(score, words)
        return np.mean(scores)
    
class DiversityEvaluator:
    def __init__(self, device="cpu", stopwords=[], target_model=None):
        self.device = device
        self.base_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.lemmatizer = WordNetLemmatizer()
        self.target_model = target_model
        self.stopwords = stopwords
        self.word2processed = {} 
        self.vectorByTopics = []
        for _ in range(self.target_model.latent_size):
            self.vectorByTopics.append({})

    def fit_embeddings(self, data_generator):
        self.base_model.eval()
        self.target_model.eval()
        for i, sample in enumerate(data_generator):
            input_ids = sample[0].to(self.device)
            attention_masks = sample[1].to(self.device)
            words_mask = sample[2].to(self.device)

            with torch.no_grad():
                _, _, _, _, _, kw = self.target_model(input_ids, attention_masks, input_ids)
                features = self.base_model(input_ids, attention_masks, output_hidden_states=True, output_attentions=True)
                local_embeddings = features.hidden_states[1]
                kw = kw * words_mask.unsqueeze(2)

                for sent_idx, input_id in enumerate(input_ids):
                    previous_token = "[CLS]"
                    previous_topic = kw[sent_idx][0].cpu().data.numpy()
                    previous_embeddings = local_embeddings[sent_idx][0].cpu().data.numpy()
                    c = 1
                    temp = []
                    for word_idx, current_token in enumerate(self.tokenizer.convert_ids_to_tokens(input_ids[sent_idx])):
                        if word_idx == 0:
                            continue
                        if current_token.startswith("##"):
                            previous_token += current_token.replace("##", "")
                            previous_topic += kw[sent_idx][word_idx].cpu().data.numpy()
                            previous_embeddings += local_embeddings[sent_idx][word_idx].cpu().data.numpy()
                            c += 1
                        else:
                            previous_topic = previous_topic/c
                            previous_embeddings = previous_embeddings/c
                            if previous_token not in ["[SEP]", "[PAD]", "[CLS]"]:
                                if previous_token not in self.word2processed:
                                    self.word2processed[previous_token] = self.lemmatizer.lemmatize(previous_token)
                                processed_token = self.word2processed[previous_token]
                                term = {"token": processed_token, "dist": previous_topic, "embeddings": previous_embeddings}
                                for k in range(self.target_model.latent_size):
                                    try:
                                        self.vectorByTopics[k][term['token']]["embeddings"] += term['embeddings'] * term['dist'][k]
                                        self.vectorByTopics[k][term['token']]["weight"] += term['dist'][k]
                                    except:
                                        self.vectorByTopics[k][term['token']] = {}
                                        self.vectorByTopics[k][term['token']]["embeddings"] = term['embeddings'] * term['dist'][k]
                                        self.vectorByTopics[k][term['token']]["weight"] = term['dist'][k]
                            previous_token = current_token
                            previous_topic = kw[sent_idx][word_idx].cpu().data.numpy()
                            previous_embeddings = local_embeddings[sent_idx][word_idx].cpu().data.numpy()
                            c = 1

    def diversity_score(self, topic_words = []):
        model_output = {"topics":[]}
        for i in range(self.target_model.latent_size):
            temp = [(w, self.vectorByTopics[i][w]['weight']) for w in topic_words[i]]
            topic = [(item[0], self.vectorByTopics[i][item[0]]["embeddings"]/self.vectorByTopics[i][item[0]]["weight"]) for item in temp]
            model_output["topics"].append(topic)
            print('topic {index}: {words}'.format(index = i, words=[w[0] for w in topic]))
        dcscore = centroid_score(model_output, 10)
        return dcscore
        
if __name__=="__main__":
    evaluator = CoherenceEvaluator()
    score = evaluator.coherence_score([["cake", "apple", "banana", "cherry", "chocolate"],["cake", "apple", "banana", "cherry", "chocolate"]])
    print(score)
    