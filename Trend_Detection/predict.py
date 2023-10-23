from Trend_Detection.train import Trend_Detection_Model
from keras.utils import pad_sequences
from utils import remove_stopwords
import numpy as np


class Trending_Data_Extractor():

  def __init__(self,
               embedding_matrix,
               model_checkpoint_path,
               text2sequence_tokenizer,
               max_length):
        
        trend_detection_model = Trend_Detection_Model(embedding_matrix=embedding_matrix,
                                                      vocab_size=len(text2sequence_tokenizer.word_index)+1)
        self.trend_detection_model = trend_detection_model.build_model(input_length=max_length)
        self.max_length = max_length
        self.trend_detection_model.load_weights(model_checkpoint_path)
        self.text2sequence_tokenizer = text2sequence_tokenizer

  def __feature_extrction(self, array_texts):
    array_texts = np.array(list(map(remove_stopwords, array_texts)))
    feature = self.text2sequence_tokenizer.texts_to_sequences(array_texts)
    return pad_sequences(feature, padding='post', maxlen=self.max_length)

  def __classify(self, text_sequence, threshold=0.3, batch_size=256):
    prediction_prob = self.trend_detection_model.predict(text_sequence, verbose=1, batch_size=batch_size)
    predictions = np.where(prediction_prob >= threshold, True, False)
    return predictions.ravel()

  def predict(self, df=None, numpy_array_texts=None, text_column='text', threshold=0.3, batch_size=256):
    if not df is None:
      array_texts = df[text_column].to_numpy()
    elif not numpy_array_texts is None:
      array_texts = numpy_array_texts
    else:
      raise Exception('------ No input ------')
    feature = self.__feature_extrction(array_texts)
    labels = self.__classify(feature, threshold=threshold, batch_size=batch_size)
    if not df is None:
      return df[labels], labels
    else:
      return numpy_array_texts[labels], labels
    