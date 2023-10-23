from utils import generate_class_weight, remove_stopwords
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import pickle
import torch
import tqdm


class DataProcessor():

  def __init__(self,
               Labeled_Dataset):
        self.data = Labeled_Dataset


  def split_data(self, split_ratio=0.2, validation_spilt=None):
    X_train, X_test, y_train, y_test = train_test_split(self.data.drop(['Label'], axis=1), self.data['Label'], test_size=split_ratio,
                                                        random_state=0, stratify=self.data[['Label','Topic']])


    if not validation_spilt is None:
      X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                      test_size=validation_spilt,
                                                                      random_state=0,
                                                                      stratify=pd.concat([X_train['Topic'], y_train], axis=1))
      y_train = np.asarray(y_train).astype('int')
      y_test = np.asarray(y_test).astype('int')
      y_validation = np.asarray(y_validation).astype('int')
      return X_train.reset_index(drop=True), X_test.reset_index(drop=True), X_validation.reset_index(drop=True), y_train, y_test, y_validation

    else:
      y_train = np.asarray(y_train).astype('int')
      y_test = np.asarray(y_test).astype('int')
      return X_train, X_test, y_train, y_test

  def sampeler(self, X_train, y_train, strategy='Class_weight'):
    if strategy=='Over':
      sampler = RandomOverSampler(random_state=0)
    elif strategy=='Under':
      sampler = RandomUnderSampler(random_state=0)
    elif strategy=='Class_weight':
      return generate_class_weight(y_train)

    return sampler.fit_resample(X_train, y_train)

  def __text2sequence(self, X_train, y_train, X_test, y_test, topic_array_test, X_validation=None, y_validation=None, RemoveStopWords=True, text_col='text'):

    x_train_text = X_train[text_col].to_numpy()
    x_test_text = X_test[text_col].to_numpy()

    if RemoveStopWords:
      print('--------------- Removing stopwords from train dataset...')
      x_train_text = np.array(list(map(remove_stopwords, x_train_text)))
      indicies = np.argwhere(x_train_text=='').ravel()
      # print(indicies)
      x_train_text = np.delete(x_train_text, indicies)
      y_train = np.delete(y_train, indicies)
      print('--------------- Removing stopwords from test dataset...')
      x_test_text = np.array(list(map(remove_stopwords, x_test_text)))
      indicies = np.argwhere(x_test_text=='').ravel()
      # print(indicies)
      x_test_text = np.delete(x_test_text, indicies)
      y_test = np.delete(y_test, indicies)
      topic_array_test = np.delete(topic_array_test, indicies)

    if not X_validation is None:
      x_validation_text = X_validation[text_col].to_numpy()
      indicies = np.argwhere(x_validation_text=='').ravel()
      # print(indicies)
      x_validation_text = np.delete(x_validation_text, indicies)
      y_validation = np.delete(y_validation, indicies)
      if RemoveStopWords:
        print('--------------- Removing stopwords from Validation dataset...')
        x_validation_text = np.array(list(map(remove_stopwords, x_validation_text)))

    max_length = len(max(x_train_text, key=len))
    tokenizer = Tokenizer()
    print('--------------- fit_on_texts() called...')
    tokenizer.fit_on_texts(x_train_text)

    print('--------------- train2sequence...')
    X_train_sequence = tokenizer.texts_to_sequences(x_train_text)
    X_train_sequence = pad_sequences(X_train_sequence, padding='post', maxlen=max_length)

    print('--------------- test2sequence...')
    X_test_sequence = tokenizer.texts_to_sequences(x_test_text)
    X_test_sequence = pad_sequences(X_test_sequence, padding='post', maxlen=max_length)

    if not X_validation is None:
      print('--------------- validation2sequence...')
      X_validation_sequence = tokenizer.texts_to_sequences(x_validation_text)
      X_validation_sequence = pad_sequences(X_validation_sequence, padding='post', maxlen=max_length)
      return X_train_sequence, y_train, X_test_sequence, y_test, X_validation_sequence, y_validation, tokenizer, topic_array_test

    else:
      return X_train_sequence, y_train, X_test_sequence, y_test, tokenizer, topic_array_test


  def __Bert_Embedding(self, bert_model, bert_tokenizer, text2sequence_tokenizer):
    word_index=text2sequence_tokenizer.word_index
    vocab_size=len(word_index)+1
    embedding_matrix = np.zeros((vocab_size, 768))

    for word, i in tqdm.tqdm(word_index.items()):
      if i > vocab_size:
          continue
      try:
        encoded_input = bert_tokenizer(word, return_tensors='pt')
        if torch.cuda.is_available():
          encoded_input.to("cuda:0")
          embedding_vector = bert_model(**encoded_input)[-1][0].cpu().detach().numpy()
        else:
          embedding_vector = bert_model(**encoded_input)[-1][0].detach().numpy()
        # embedding_vector = bert_model.encode(word)
      except:
        embedding_vector=None
      if embedding_vector is not None:
          embedding_matrix[i]=embedding_vector
    return embedding_matrix

  def feature_extractor(self,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        topic_array_test,
                        bert_model,
                        bert_tokenizer,
                        X_validation=None,
                        y_validation=None,
                        RemoveStopWords=True,
                        sequence_tokenizer_saving_path='',
                        embedding_matrix_saving_path=''):

    print('------- text2sequence() Called...')
    if not X_validation is None:
      X_train_sequence, y_train, X_test_sequence, y_test, X_validation_sequence, y_validation, text2sequence_tokenizer, topic_array_test = self.__text2sequence(X_train,
                                                                                                                                                                y_train,
                                                                                                                                                                X_test,
                                                                                                                                                                y_test,
                                                                                                                                                                topic_array_test,
                                                                                                                                                                X_validation=X_validation,
                                                                                                                                                                y_validation=y_validation,
                                                                                                                                                                RemoveStopWords=RemoveStopWords)
    else:
      X_train_sequence, y_train, X_test_sequence, y_test, text2sequence_tokenizer, topic_array_test = self.__text2sequence(X_train,
                                                                                                                           y_train,
                                                                                                                           X_test,
                                                                                                                           y_test,
                                                                                                                           topic_array_test,
                                                                                                                           RemoveStopWords=RemoveStopWords)

    print('------- Bert_Embedding() Called...')
    embedding_matrix = self.__Bert_Embedding(bert_model, bert_tokenizer, text2sequence_tokenizer)

    with open(sequence_tokenizer_saving_path+'/text2sequence_tokenizer.pkl', 'wb') as handle:
        pickle.dump(text2sequence_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(embedding_matrix_saving_path+'/embedding_matrix_trend_detection.pkl', 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not X_validation is None:
      return  X_train_sequence, y_train, X_test_sequence, y_test, X_validation_sequence, y_validation, embedding_matrix, topic_array_test
    else:
      return X_train_sequence, y_train, X_test_sequence, y_test, embedding_matrix, topic_array_test