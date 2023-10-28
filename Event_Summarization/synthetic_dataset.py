from torch.utils.data import Dataset
from operator import itemgetter
from itertools import groupby
from utils import *
import pandas as pd
import numpy as np
import pickle
import random
import torch
import glob
import ast

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
MAXLEN          = 768

class Synthetic_Dataset(Dataset):
    def __init__(self,
                 NE_paths,
                 df_paths,
                 start_date,
                 end_date,
                 tokenizer,
                 context_length=2,
                 max_occurrnece=30,
                 randomize=False):
        
        data = self.__build_balance_dataset(NE_paths,
                                            df_paths,
                                            start_date,
                                            end_date,
                                            context_length=context_length,
                                            max_occurrnece=max_occurrnece)
        text, keywords = [], []
        for _, v in data.items():
          text.append(v[0])
          keywords.append(v[1])

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.text      = text
        self.keywords  = keywords

    #---------------------------------------------#

    @staticmethod
    def join_keywords(keywords, randomize=False):
        N = len(keywords)

        #random sampling and shuffle
        if randomize:
            M = random.choice(range(N+1))
            keywords = keywords[:M]
            random.shuffle(keywords)

        return ','.join(keywords)

    #---------------------------------------------#

    def __len__(self):
        return len(self.text)

    #---------------------------------------------#

    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        kw = self.join_keywords(keywords, self.randomize)

        input = SPECIAL_TOKENS['bos_token'] + kw + SPECIAL_TOKENS['sep_token'] + self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=MAXLEN,
                                        padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}

    def __build_balance_dataset(self, NE_paths, df_paths, start_date, end_date, context_length=1, max_occurrnece=30):
      end_of_windows = pd.date_range(start=start_date, end=end_date, freq='1D', tz='UTC').strftime('%Y-%m-%d / %H:%M:%S')
      dfs_segmented_data = []

      for filepath in sorted(glob.iglob(df_paths + '/*.csv')):
        dfs_segmented_data.append(pd.read_csv(filepath))
      with open(NE_paths, 'rb') as f:
        all_NameEntities = pickle.load(f)

      occurrence = {}

      for entity, occ in all_NameEntities.items():
        if entity[0] in occurrence:
          for id in occ:
            occurrence[entity[0]].append((id, entity[-1]))
        else:
          occurrence[entity[0]] = []
          for id in occ:
            occurrence[entity[0]].append((id, entity[-1]))

      df_index = []
      for key, value in occurrence.items():
        if len(value) >= 1:
          if len(value) <= max_occurrnece:
            df_index.extend(value)
          else:
            selection = np.random.choice(range(0,len(value)), size=max_occurrnece, replace=False)
            df_index.extend([value[i] for i in selection])

          for entity, occ in occurrence.items():
            if len(occ) >= 1:
              occurrence[entity] = list(set(occ).difference(value))

      sorter = sorted(df_index, key=itemgetter(1))
      grouper = groupby(sorter, key=itemgetter(1))
      df_index = {k: list(map(itemgetter(0), v)) for k, v in grouper}

      result = {}
      id=1
      for key, value in df_index.items():
        df = dfs_segmented_data[end_of_windows.get_loc(key)].iloc[value,:]
        for tweet, hashtags in df[['text', 'hashtags']].itertuples(index=False):
          tweet = tweet.replace('ج .  ا', 'جمهوری اسلامی ایران')
          hashtags = ast.literal_eval(hashtags)
          if type(hashtags) == list:
            tweet = remove_useless_hashtags(tweet, hashtags)
          tweet = remove_punctuation(tweet)
          tokens = []
          for token in tweet.split():
            if token in occurrence:
              tokens.append(token)
          features = []
          for entity in set(tokens):
            features = np.union1d(features, extract_context(tweet, entity, context_length=context_length))
          features = refine_contexts(tweet, features)
          if len(features)>1:
            result[id] = [tweet, features]
            id+=1
      return result

