from sklearn.metrics.pairwise import pairwise_distances
from utils import *
import pandas as pd
import numpy as np
import pickle
import torch
import ast

class Extract_Entities():
    def __init__(self,
               dataset,
               embedding_model_path,
               NER_model_path,
               trending_data_extractor=None
               ):
        self.dataset = dataset
        self.bert_embedding_model, self.bert_tokenizer = get_bert(embedding_model_path)
        if torch.cuda.is_available():
          self.bert_embedding_model = self.bert_embedding_model.to("cuda:0")
        self.NER_model_path = NER_model_path
        if not trending_data_extractor is None:
          self.trending_data_extractor = trending_data_extractor
        self.NER_model, self.NER_tokenizer, self.NER_labels = get_NER_model(self.NER_model_path)


    def segment_dataframe(self, start_date, end_date, window_size, window_length):
      windowed_df = []
      end_of_windows = pd.date_range(start=start_date, end=end_date, freq=window_size, tz='UTC')
      for i in range(len(end_of_windows)):
        if i != len(end_of_windows)-1 :
          window = self.dataset[(self.dataset['datetime'] >= end_of_windows[i]) & (self.dataset['datetime'] < end_of_windows[i+1])]
        else:
          continue
        window = sample_df(window, window_length)
        windowed_df.append(window)
      return windowed_df

    def text2entity(self, df):
      NER = NER_detection(df['text'], self.NER_model, self.tokenizer, self.labels)
      hashtags = get_unique_hashtags(df)
      NER.extend(hashtags)
      return clean_entities(NER, punctuations, stopwords)

    def generate_occurrence_matrix(self, df, list_entities, NE_occurrence_dict_saving_path=None, date=None):
      occurrence = np.zeros(shape=(len(list_entities),len(df['text'])), dtype=np.int16)
      for i, text in enumerate(df['text']):
        for j, nameEntity in enumerate(list_entities):
          occurrence[j,i] = calculate_cooccurrence(text, nameEntity)

      list_entities, occurrence = remove_useless_entities(list_entities, occurrence)
      if (not NE_occurrence_dict_saving_path is None) and (not date is None):
        self.__save_nameEntites_occurrence(occurrence,
                                           np.array(list_entities),
                                           date = date,
                                           NE_occurrence_dict_path= NE_occurrence_dict_saving_path)
      return list_entities, occurrence

    def generate_distance_matrix(self, feature_matrix, distance_metric='cosine'):
      return pairwise_distances(feature_matrix, metric=distance_metric)

    def generate_embedding_matrix(self, df, occurrence_matrix, list_entities, context_selector=['all', 3]):
      embedding = np.zeros(shape=(len(list_entities),768), dtype=np.int16)
      for i, nameEntity in enumerate(list_entities):
          indicies = np.where(occurrence_matrix[i] >= 1)[0]
          selected_df = df.iloc[indicies]

          if context_selector[0] == 'all':
            texts = []
            for tweet, hashtags in selected_df[['text', 'hashtags']].itertuples(index=False):
              hashtags = ast.literal_eval(hashtags)
              if type(hashtags) == list:
                tweet = remove_useless_hashtags(tweet, hashtags)
              texts.append(tweet)
            texts = np.vectorize(extract_context)(np.array(texts, str), nameEntity, context_selector[-1])

          else:
            temp = select_tweets_embedding(selected_df, k=context_selector[-1], column=context_selector[0])
            texts = []
            for tweet, hashtags in temp[['text', 'hashtags']].itertuples(index=False):
              hashtags = ast.literal_eval(hashtags)
              if type(hashtags) == list:
                tweet = remove_useless_hashtags(tweet, hashtags)
              texts.append(tweet)

          text = ' '.join(texts)
          text = remove_stopwords(text)
          if len(text) != 0 :
            encoded_input = self.bert_tokenizer(text, return_tensors='pt')
            if torch.cuda.is_available():
              encoded_input.to("cuda:0")
              embedding[i] = self.bert_embedding_model(**encoded_input)[-1][0].cpu().detach().numpy()
            else:
              embedding[i] = self.bert_embedding_model(**encoded_input)[-1][0].detach().numpy()

      return embedding

    def __save_nameEntites_occurrence(self, occurrence_matrix, list_entities, date, NE_occurrence_dict_path=None):
      try:
        with open(NE_occurrence_dict_path, 'rb') as handle:
          NE_occurrence_dict = pickle.load(handle)
      except:
        NE_occurrence_dict = {}

      for i, nameEntity in enumerate(list_entities):
        indicies = np.where(occurrence_matrix[i] >= 1)[0]
        NE_occurrence_dict[(nameEntity, date)] = indicies

      with open(NE_occurrence_dict_path, 'wb') as handle:
        pickle.dump(NE_occurrence_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
      return