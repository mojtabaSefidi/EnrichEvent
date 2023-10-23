from Event_Summarization.synthetic_dataset import Synthetic_Dataset
from sklearn.cluster import DBSCAN
from utils import *
import pandas as pd
import numpy as np
import scipy
import torch
import ast

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

class Summary_Generator():
    def __init__(self,
                 summarization_model_path,
                 embedding_model_path,
                 gpt_model_path):
        
        self.gpt_tokenizer = get_gpt2_tokenizer(gpt_model_path=gpt_model_path,
                                                special_tokens=SPECIAL_TOKENS)
        
        self.summerization_model = get_gpt2_model(tokenizer = self.gpt_tokenizer,
                                                  gpt_model_path=gpt_model_path, 
                                                  special_tokens=SPECIAL_TOKENS,
                                                  load_model_path=summarization_model_path)
        
        self.bert_embedding_model, self.bert_tokenizer = get_bert(embedding_model_path)
        if torch.cuda.is_available():
          self.bert_embedding_model = self.bert_embedding_model.to("cuda:0")


    def __generate_phrase(self, event_path, context_length=1):
      df = pd.read_csv(event_path+'/final_df.csv')
      phrase = []
      with open(event_path + '/all_NameEntities.txt', 'r', encoding="utf-8") as f:
        name_entities = f.read().splitlines()
        name_entities = [entity.split(':')[0].strip() for entity in name_entities]
      for text, hashtags in df[['text', 'hashtags']].itertuples(index=False):
        text = text.replace('ج .  ا', 'جمهوری اسلامی ایران')
        hashtags = ast.literal_eval(str(hashtags))
        if type(hashtags) == list:
          text = remove_useless_hashtags(text, hashtags)
        text = remove_stopwords(text)
        features = []
        for entity in name_entities:
          # print(entity, text, extract_context(text, entity, context_length=context_length))
          features = np.union1d(features, extract_context(text, entity, context_length=context_length))
        features = refine_contexts(text, features)
        phrase = np.union1d(phrase, features)
      return phrase[phrase != '']

    def __data2keyphrase(self, list_phrases):
      key_phrases = {}
      for phrase in list_phrases:
        phrase_processed = remove_stopwords(phrase)
        encoded_input = self.bert_tokenizer(phrase_processed, return_tensors='pt')
        if torch.cuda.is_available():
          encoded_input.to("cuda:0")
          embedding_vector = self.bert_embedding_model(**encoded_input)[-1][0].cpu().detach().numpy()
        else:
          embedding_vector = self.bert_embedding_model(**encoded_input)[-1][0].detach().numpy()
        if phrase not in key_phrases:
          key_phrases[phrase] = embedding_vector
        else:
          key_phrases[phrase] = np.mean([key_phrases[phrase], embedding_vector], axis=0)
      return key_phrases

    def __keyphrase_clustering(self, key_phrases, k=15, eps=0.5, min_samples=2):
      feature = np.array(list(key_phrases.values()))
      clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(feature)
      clusters = {}
      for index, label in enumerate(clustering.labels_):
        if label != -1:
          if label in clusters:
            clusters[label].append(list(key_phrases.keys())[index])
          else:
            clusters[label] = [list(key_phrases.keys())[index]]
      clusters = {key: value for key, value in sorted(clusters.items())}
      return list(clusters.values())[:k]

    def __find_closest(self, clusters, key_phrases):
      result = []
      for cluster in clusters:
        embedding = list(map(key_phrases.get, cluster))
        center = np.mean(embedding, axis=0)
        distance = np.inf
        index = None
        for i, embedd in enumerate(embedding):
          if scipy.spatial.distance.cosine(center, embedd) < distance:
            index = i
        if not index is None:
          result.append(cluster[index])
      return result

    def __predict(self, keywords, number_prediction=2, min_length=50, max_length=200):
      kw = Synthetic_Dataset.join_keywords(keywords, randomize=False)
      prompt = SPECIAL_TOKENS['bos_token'] + kw + SPECIAL_TOKENS['sep_token']
      generated = torch.tensor(self.gpt_tokenizer.encode(prompt)).unsqueeze(0)
      device = torch.device("cuda")
      generated = generated.to(device)
      self.summerization_model.eval()

      sample_outputs = self.summerization_model.generate(generated,
                                                        do_sample=True,
                                                        min_length=min_length,
                                                        max_length=max_length,
                                                        top_k=30,
                                                        top_p=0.7,
                                                        temperature=0.9,
                                                        repetition_penalty=2.0,
                                                        num_return_sequences=number_prediction)
      result = []
      for sample_output in sample_outputs:
        text = self.gpt_tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(','.join(keywords))
        result.append(text[a:])
      return np.array(result)

    def __find_closest_summary(self, event_path, list_summary):
      df = pd.read_csv(event_path + '/final_df.csv')
      result = {}
      for i, summary in enumerate(list_summary):
        similarity = []
        encoded_input = self.bert_tokenizer(remove_stopwords(summary), return_tensors='pt')
        if torch.cuda.is_available():
          encoded_input.to("cuda:0")
          summary_embedding = self.bert_embedding_model(**encoded_input)[-1][0].cpu().detach().numpy()
        else:
          summary_embedding = self.bert_embedding_model(**encoded_input)[-1][0].detach().numpy()
        for text, hashtags in df[['text', 'hashtags']].itertuples(index=False):
          text = text.replace('ج .  ا', 'جمهوری اسلامی ایران')
          hashtags = ast.literal_eval(str(hashtags))
          if type(hashtags) == list:
            text = remove_useless_hashtags(text, hashtags)
          encoded_input = self.bert_tokenizer(remove_stopwords(text), return_tensors='pt')
          if torch.cuda.is_available():
            encoded_input.to("cuda:0")
            text_embedding = self.bert_embedding_model(**encoded_input)[-1][0].cpu().detach().numpy()
          else:
            text_embedding = self.bert_embedding_model(**encoded_input)[-1][0].detach().numpy()
          similarity.append(round(1 - scipy.spatial.distance.cosine(summary_embedding, text_embedding), 4))

        result[i] = (similarity, round(np.mean(similarity), 5))
      result = dict(sorted(result.items(), key=lambda item: item[1][1], reverse=True))
      return result
    def generate_event_summary(self,
                               event_path,
                               eps=0.4,
                               min_samples=1,
                               num_clusters=3,
                               number_prediction=5,
                               min_length=30,
                               max_length=100,
                               context_length=3):

      print('   2.1) Retrieve key phrases...')
      list_phrases = self.__generate_phrase(event_path,
                                            context_length=context_length)
      # print(list_phrases)
      key_phrases = self.__data2keyphrase(list_phrases)
      print('   2.2) Produce clusters...')
      clusters = self.__keyphrase_clustering(key_phrases,
                                             k=num_clusters,
                                             eps=eps,
                                             min_samples=min_samples)
      # print(clusters)
      print('   2.3) Filter clusters...')
      keywords = self.__find_closest(clusters,
                                    key_phrases)
      # print(keywords)
      print('   2.4) Generate summerizations...')
      list_summary = self.__predict(keywords=keywords,
                                    number_prediction=number_prediction,
                                    min_length=min_length,
                                    max_length=max_length)

      print('   2.5) Filter summaries and return the best one...')
      dict_summary = self.__find_closest_summary(event_path, list_summary)
      indicies = list(dict_summary.keys())[0:num_clusters]
      list_summary = [summary.replace('\u200c', ' ') for summary in list_summary[indicies]]
      return dict_summary, list_summary
