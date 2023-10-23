from transformers import AutoTokenizer, AutoConfig, TFAutoModelForTokenClassification, BertTokenizer, BertModel, AutoModelForPreTraining
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import linear_sum_assignment
from collections import Counter
from tensorflow import keras
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow
import itertools
import operator
import hdbscan
import pickle
import torch
import glob
import math
import ast
import re

with open('Essential_Files/stopwords.txt', 'r') as f:
  stopwords = list(set(f.read().splitlines()))
  stopwords_dict = Counter(stopwords)

with open('Essential_Files/punctuation.txt', 'r') as f:
  punctuations = list(set(f.read().splitlines()))

def generate_class_weight(y_train):
    class_weights = {}
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight = "balanced", classes= classes, y=y_train)
    for i in range(len(classes)):
        class_weights[classes[i]] = weights[i]
    return class_weights

def filter_by_index(X_test_sequence, y_test, topic_indices):
    return X_test_sequence[topic_indices], y_test[topic_indices]

def topic2index(topic_array):
  result = {}
  keys = np.unique(topic_array)
  for key in keys:
    result[key] = np.argwhere(topic_array==key).ravel()
  return result

def remove_punctuation(text):
  text = text.replace(' .', '.')
  for punc in punctuations:
    text = text.replace(punc, ' ')
  text = re.sub(r'\s*[A-Za-z]+\b', ' ' , text).strip()
  return re.sub(' +', ' ', text)

def remove_stopwords(text):
  text = text.replace('  ', ' ')
  text = text.replace(' .', '.')
  for punc in punctuations:
    text = text.replace(punc, ' ')
  text = re.sub(r'\s*[A-Za-z]+\b', ' ' , text).strip()
  text = re.sub(' +', ' ', ' '.join([word for word in text.split() if word not in stopwords_dict]))
  return re.sub(' +', ' ', text)

def get_bert(path):
  bert_tokenizer = BertTokenizer.from_pretrained(path)
  bert_model = BertModel.from_pretrained(path)
  return bert_model, bert_tokenizer

def get_earlystoping(metric='val_Precision', patience=5, mode='auto'):
  early_stopping = keras.callbacks.EarlyStopping(monitor=metric,
                                                 verbose=1,
                                                 patience=patience,
                                                 mode=mode,
                                                 restore_best_weights=True)
  return early_stopping

def get_callback(checkpoint_path, sample_per_epoch, frequency=5):
  cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                verbose=1,
                                                save_weights_only=True,
                                                save_freq = frequency*sample_per_epoch)
  return cp_callback

def get_classification_metrics():
  metrics = [keras.metrics.TruePositives(name='TP'),
            keras.metrics.FalsePositives(name='FP'),
            keras.metrics.TrueNegatives(name='TN'),
            keras.metrics.FalseNegatives(name='FN'),
            keras.metrics.BinaryAccuracy(name='Accuracy'),
            keras.metrics.Precision(name='Precision'),
            keras.metrics.Recall(name='Recall'),
            keras.metrics.AUC(name='AUC'),
            keras.metrics.AUC(name='PRC', curve='PR')]
  return metrics

def read_dataset(dataset_path,
                 datetime_column,
                 hashtag_column=None):
  
  dataset = pd.read_csv(dataset_path)
  dataset[datetime_column] = pd.to_datetime(dataset[datetime_column], utc=True)
  dataset[datetime_column] = dataset[datetime_column].dt.tz_convert('UTC')
  if not hashtag_column is None:
    dataset[hashtag_column] = dataset[hashtag_column].fillna('-1')
    dataset[hashtag_column] = dataset[hashtag_column].replace({"Without_Hashtag": "-1"})
  else:
    dataset[hashtag_column] = '-1'
  return  dataset


# event_detection

def get_all_hashtags(df, hashtag_col='hashtags', no_hashtag='-1'):
  x = df[df.loc[:,hashtag_col]!=no_hashtag][hashtag_col].apply(ast.literal_eval)
  return list(itertools.chain.from_iterable(list(x)))

def sample_df(df, window_length):
  if len(df) <= window_length:
    return df
  df = df.drop_duplicates(subset=['text']).sort_values(by='datetime').reset_index(drop=True)
  occurrence = {}
  frequency = {}
  indices = df.index
  texts = df['text'].to_numpy()
  i=0
  total = 0
  for text in texts:
    words = text.split(' ')
    total += len(words)
    for word in set(words):
      if word in frequency:
        occurrence[word].append(indices[i])
        frequency[word] = frequency[word] + words.count(word)

      else:
        occurrence[word] = [indices[i]]
        frequency[word] = words.count(word)
    i+=1
  percentage = {}
  for key, value in frequency.items():
    percentage[key] = round(value / total, 6)
  percentage = dict(sorted(percentage.items(), key=lambda item: item[1], reverse=True))

  result = np.array([], int)
  for key, value in percentage.items():
    if value*window_length >= 0.3 and value*window_length <= 0.5:
      n_selection = 1
    else:
      n_selection = round(value*window_length)
    if n_selection != 0:
      occurrence[key] = np.array(occurrence[key], int)[~np.isin(occurrence[key], result)]
      if len(occurrence[key]) >= n_selection:
        selection = np.random.choice(occurrence[key], n_selection, replace=False)
        result = np.union1d(result, selection)
        occurrence[key] = occurrence[key][~np.isin(occurrence[key], selection)]
      else:
        result = np.union1d(result, occurrence[key])
        del occurrence[key]
        del percentage[key]
    if len(result)>= window_length:
      return df.iloc[sorted(result)].reset_index(drop=True)
  for key, value in percentage.items():
    if len(occurrence[key]) >= 1:
      selection = np.random.choice(occurrence[key], 1, replace=False)
      result = np.union1d(result, selection)
      occurrence[key] = np.array(occurrence[key], int)[~np.isin(occurrence[key], selection)]
    else:
      del occurrence[key]
      del percentage[key]
    if len(result)>= window_length:
      return df.iloc[sorted(result)].reset_index(drop=True)

  return df.iloc[sorted(result)].reset_index(drop=True)


def get_unique_hashtags(df, hashtag_col='hashtags', no_hashtag='-1'):
  x = df[df.loc[:,hashtag_col]!=no_hashtag][hashtag_col].apply(ast.literal_eval)
  return list(set(itertools.chain.from_iterable(list(x))))

def clean_entities(list_entities,
                   punctuations,
                   stopwords):
  entities = list(i[0] for i in set(list(itertools.chain.from_iterable(list([x for x in list_entities if x])))) if not '#' in i[0])
  entities = [NER for NER in entities if len(NER)>=3]
  entities = set(NER for NER in entities if not NER.upper().isupper())
  return list(entities.difference(set.union(set(punctuations), set(stopwords))))

def calculate_cooccurrence(text, nameEntity):
  if '_' in text:
    text = text.replace('_', ' ').strip()
  return text.lower().split(' ').count(nameEntity)

def remove_useless_entities(entities, occurrence):
  useless_index = np.where(occurrence.sum(1) == 0)[0]
  occurrence = np.delete(occurrence, useless_index, 0)
  entities = list(x for i, x in enumerate(list(entities)) if i not in useless_index)
  return entities, occurrence

def remove_useless_hashtags(text, hashtags):
  hashtags = [hashtag.replace('_' ,' ') for hashtag in hashtags]
  hashtags_dict = {}
  for hashtag in hashtags:
    hashtags_dict[hashtag] = text.rfind(hashtag)
  hashtags_dict = dict(sorted(hashtags_dict.items(), key=lambda item: item[1], reverse=True))
  for hashtag in hashtags_dict.keys():
    start = text.rfind(hashtag)
    end = start + len(hashtag)
    if hashtag.lower() != hashtag:
      text = (text[:start] + text[end+1:]).strip()
    elif end == len(text):
      text = (text[:start] + text[end+1:]).strip()
  return text.strip()

def extract_context(text, Name_Entity, context_length=1):
  tokens = text.split()
  result = []
  for i in range(len(tokens)):
    if tokens[i] == Name_Entity:
      back = i
      front = len(tokens)-i-1
      context_back = context_length
      context_front = context_length

      if context_back > back and front >= context_front + context_back - back:
        context_front = context_front + context_back - back
        context_back = back
        start = i - context_back
        end = i + context_front + 1
        result.append(' '.join(tokens[start:end]))

      elif context_front > front and back >= context_back + context_front - front:
        context_back = context_back + context_front - front
        context_front = front
        start = i - context_back
        end = i + context_front + 1
        result.append(' '.join(tokens[start:end]))

      elif (context_front > front and back < context_back + context_front - front) or (context_back > back and front < context_front + context_back - back):
        result.append(' '.join(tokens[:]))

      else:
        start = i - context_back
        end = i + context_front + 1
        result.append(' '.join(tokens[start:end]))
  
  return result

def select_tweets_embedding(df, k, column='retweet_count'):
  if len(df) <= k:
    return df
  else:
    indicies = df.sort_values(column, ascending=False).index
    result = np.union1d([], indicies[:math.ceil(k/2)])
    indicies = indicies[math.ceil(k/2):]
    result = np.union1d(result, np.random.choice(indicies, k-math.ceil(k/2), replace=False))
    return df[df.index.isin(result)]
  
def get_NER_model(NER_model_path):
  config = AutoConfig.from_pretrained(NER_model_path)
  tokenizer = AutoTokenizer.from_pretrained(NER_model_path)
  model = TFAutoModelForTokenClassification.from_pretrained(NER_model_path)
  labels = list(config.label2id.keys())
  return model, tokenizer, labels

def NER_detection(df_texts, model, tokenizer, labels):
    if not model or not tokenizer or not labels:
        raise Exception('Something wrong has been happened!')
    output_predictions = []
    i = 1
    for text in df_texts:
        if i % 1000 == 0:
          print(str(i) + ' samples passed.')
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        inputs = tokenizer.encode(text, return_tensors="tf")
        outputs = model(inputs)[0]
        predictions = tensorflow.argmax(outputs, axis=2)
        predictions = [(token, labels[prediction]) for token, prediction in zip(tokens, predictions[0].numpy()) if labels[prediction] != 'O']
        output_predictions.append(predictions)
        i+=1
    return output_predictions

def find_common_entities(cluster1_dict, cluster2_dict):
  key = list(itertools.product(list(cluster1_dict.keys()), list(cluster2_dict.keys())))
  product_values = list(itertools.product(list(cluster1_dict.values()), list(cluster2_dict.values())))
  value = list(map(lambda i:len(find_instersection(product_values[i])), range(0, len(product_values))))
  return dict(zip(key,value))

def find_instersection(tuple_entity):
  return np.intersect1d(tuple_entity[0], tuple_entity[-1])

def max_bipartite_matching(common_dict):
  row_names = sorted(set(i[0] for i in common_dict.keys()))
  col_names = sorted(set(i[-1] for i in common_dict.keys()))
  G = np.array(np.zeros(shape=(len(row_names), len(col_names))), int)
  for i, row in enumerate(row_names):
    for j, col in enumerate(col_names):
      G[i,j] = common_dict.get((row, col), 0)

  row_indices, col_indices = linear_sum_assignment(G, maximize=True)
  return [((row_names[r], col_names[c])) for r, c in zip(row_indices, col_indices)]

def find_add_pair(history, tup):
  for index, l in enumerate(history):
    if tup[0] in l:
      history[index].append(tup[1])
      return history

    elif tup[1] in l:
      history[index].append(tup[0])
      return history

  return history.append([min(tup), max(tup)])

def store_dfs(segmented_dfs, end_of_windows, path):
  Path(path).mkdir(parents=True, exist_ok=True)
  for i, df in enumerate(segmented_dfs):
    df.to_csv(path+'/{0}.csv'.format(str(end_of_windows[i].strftime('%Y-%m-%d_%H:%M'))), index=False)
  return

def format_entities(name_entity_path):
  with open(name_entity_path, 'rb') as f:
    entities = pickle.load(f)
  keys = [list(group) for key,group in itertools.groupby(sorted(entities.keys(), key=lambda z: z[1]), operator.itemgetter(1))]
  return [set(next(zip(*i))) for i in keys]


def save_event_detail(cluster_chain, list_clusters, chainID, saving_path, end_of_windows):

  all_NE = np.array([])
  path = saving_path + '/event_{0}'.format(chainID+1)
  Path(path).mkdir(parents=True, exist_ok=True)

  for key in cluster_chain:
    index = end_of_windows.get_loc(key[-1])
    cluster = list_clusters[index].get(key)
    all_NE = np.union1d(all_NE, cluster)
    with open(path + '/window{0}_cluster{1}.txt'.format(index+1, key[0]), 'a') as f:
      for line in cluster:
        f.write(f"{line}\n")

  with open(path + '/all_NameEntities.txt'.format(index+1, key[0]), 'a') as f:
    for line in all_NE:
      f.write(f"{line}\n")

  return

def tweet_hashtag_retrieve(event_path, NE_paths, df_paths, end_of_windows, export_path):

  final_df = pd.DataFrame()

  dfs_segmented_data = []
  for filepath in sorted(glob.iglob(df_paths + '/*.csv')):
    dfs_segmented_data.append(pd.read_csv(filepath))

  with open(NE_paths, 'rb') as f:
    all_entities = pickle.load(f)

  for path in sorted(glob.iglob(event_path + '/window*')):
    start = path.rindex('window')+len('window')
    end = path.rfind('_')
    index = int(path[start:end])
    with open(path, 'r') as f:
      name_entities = f.read().splitlines()

    df = pd.DataFrame()
    for entity in name_entities:
      occurrence = all_entities[(entity, end_of_windows[index-1])]
      temp = dfs_segmented_data[index-1].iloc[occurrence]
      df = pd.concat([df, temp], axis=0)

    # df = df.drop_duplicates(subset=['text', 'datetime'], keep='first')
    df = df[df.duplicated(subset=['text', 'datetime'], keep=False)].drop_duplicates(subset=['text', 'datetime'], keep='first').reset_index(drop=True)
    with open(export_path + '/users_freq_time_frame.txt', 'a') as f:
      f.write(f"{df['username'].nunique()} Unique users exist in time frame {filepath[filepath.rfind('/')+1 : filepath.find('.csv')]} :\n")
      for user, freq in df['username'].value_counts().to_dict().items():
        f.write(f"{user} : {freq}\n")
      f.write("\n")

    with open(export_path + '/hashtags_freq_time_frame.txt', 'a') as f:
      all_hashtags = get_all_hashtags(df)
      f.write(f"{len(set(all_hashtags))} Unique hashtags exist in time frame {filepath[filepath.rfind('/')+1 : filepath.find('.csv')]} :\n")
      for hashtag, freq in Counter(all_hashtags).most_common():
        f.write(f"{hashtag} : {freq}\n")
      f.write("\n")

    final_df = pd.concat([final_df, df], axis=0)

  final_df = final_df.drop_duplicates(subset=['text', 'datetime'], keep='first')
  final_df = final_df.sort_values(by='datetime').reset_index(drop=True)
  if 'Unnamed: 0' in final_df.columns:
    final_df = final_df.drop('Unnamed: 0', axis=1)

  final_df.to_csv(export_path + '/final_df.csv', index=False)

  with open(export_path + '/users_freq_all.txt', 'a') as f:
    for key, value in final_df['username'].value_counts().to_dict().items():
      f.write(f"{key} : {value}\n")

  with open(export_path + '/hashtags_all.txt', 'a') as f:
    all_hashtags = get_all_hashtags(final_df)
    f.write(f"{len(set(all_hashtags))} Unique hashtags exist altogether \n")
    for hashtag, freq in Counter(all_hashtags).most_common():
      f.write(f"{hashtag} : {freq}\n")
    f.write("\n")

  with open(export_path + '/all_tweets.txt', 'a') as f:
    for tweet in final_df['text'].to_numpy():
      f.write(f"{tweet}\n")

  return

# Summarization

def refine_contexts(text, contexts):
  if len(contexts) <= 1:
    return contexts
  processed_text = text
  context_ranges = {}
  for context in contexts:
    start = processed_text.find(context)
    end = start + len(context)
    context_ranges[context] = [start, end]
  context_ranges = dict(sorted(context_ranges.items(), key=lambda item: item[1][0]))
  refined = []
  ranges = list(context_ranges.values())
  start = ranges[0][0]
  temp = ranges[0][1]
  for i in range(len(ranges)-1):
    if ranges[i+1][0] <= ranges[i][1] and ranges[i+1][0] >= ranges[i][0]:
      temp = ranges[i+1][1]
      if i+2 == len(ranges):
        refined.append(processed_text[start:temp])
    else:
      refined.append(processed_text[start:temp])
      start = ranges[i+1][0]
      temp = ranges[i+1][1]
      if i+2 == len(ranges):
        refined.append(processed_text[start:temp])
  return refined



def get_gpt2_tokenizer(gpt_model_path, special_tokens=False):
  tokenizer = AutoTokenizer.from_pretrained(gpt_model_path)
  if special_tokens:
      tokenizer.add_special_tokens(special_tokens)
  return tokenizer

def get_gpt2_model(tokenizer, gpt_model_path, special_tokens=None, load_model_path=None, UNFREEZE_LAST_N=4):
    if special_tokens:
        config = AutoConfig.from_pretrained(gpt_model_path,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else:
        config = AutoConfig.from_pretrained(gpt_model_path,
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)
    model = AutoModelForPreTraining.from_pretrained(gpt_model_path, config=config)
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
    model.cuda()
    for parameter in model.parameters():
      parameter.requires_grad = False
    for i, m in enumerate(model.transformer.h):
      if i+1 > 12 - UNFREEZE_LAST_N:
        for parameter in m.parameters():
            parameter.requires_grad = True
    for parameter in model.transformer.ln_f.parameters():
      parameter.requires_grad = True
    for parameter in model.lm_head.parameters():
      parameter.requires_grad = True
    return model

def get_hdbscan(min_cluster_size=2,
                min_samples=5,
                alpha=0.65,
                metric='precomputed'):
  
  clustering_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                   min_samples=min_samples,
                                   alpha=alpha,
                                   metric=metric)
  return clustering_model
