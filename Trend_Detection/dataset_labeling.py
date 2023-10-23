import pandas as pd
import ast

class Dataset_Labeling():

  def __init__(self,
               positive_key_phrase_path,
               spam_key_phrase_path,
               negative_key_phrase_path=None):
        
        self.positive_key_phrase = self.__read_key_phrase(positive_key_phrase_path)
        self.spam_key_phrase = self.__read_key_phrase(spam_key_phrase_path)
        if not negative_key_phrase_path is None:
          self.negative_key_phrase = self.__read_key_phrase(negative_key_phrase_path)
        else:
          self.negative_key_phrase = None

  
  def __read_key_phrase(self, file_path):
    with open(file_path, 'r') as f:
      list_key_phrase = f.read().splitlines()
    return [ast.literal_eval(key_phrase) for key_phrase in list_key_phrase]
  
  def __read_data(self, data_path):
    data = pd.read_csv(data_path, sep=',', encoding="utf-8-sig", on_bad_lines='skip')
    data['text']= data['text'].astype(str)
    return data

  def __filter(self, tokens):
    for keyphrase in self.spam_key_phrase:
      if any(token in tokens for token in keyphrase):
        return 'Spam'
    for keyphrase in self.positive_key_phrase:
      if all(token in tokens for token in keyphrase):
        return 'Positive'
    if not self.negative_key_phrase is None: 
      for keyphrase in self.negative_key_phrase:
        if all(token in tokens for token in keyphrase):
          return 'Negative'

  def __detect_labels(self, df_data):
    labels = []
    for tweet in df_data['text']:
      tokens = tweet.split(' ')
      labels.append(self.__filter(tokens))
    return labels

  def manual_labeling(self, data_path):
    data = self.__read_data(data_path)
    data['Label'] = self.__detect_labels(data)
    data.drop(data[data['Label'] == 'Spam'].index, inplace=True)
    data.loc[data['Label'] == 'Positive', ['Label']]= 1
    data.loc[data['Label'] == 'Negative', ['Label']]= 0
    return data
