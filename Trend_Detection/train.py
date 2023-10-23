from plot_utils import plot_confusion_matrix, plot_history, plot_metrics, plot_roc_curve, plot_auc_curve, plot_precision_recall_curve
from utils import filter_by_index, get_classification_metrics, get_callback, get_earlystoping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import numpy as np
import tensorflow



class Trend_Detection_Model():

  def __init__(self,
               embedding_matrix,
               vocab_size,
               class_weight=None):

        self.class_weight = class_weight
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size

  def build_model(self,
                  input_length,
                  output_dim=1,
                  filters=256,
                  kernel_size=3,
                  drop_rate=0.2,
                  units=256):

      trend_detection_model = keras.models.Sequential()
      trend_detection_model.add(keras.layers.Input(shape=(input_length)))

      #create embedding layer
      trend_detection_model.add(keras.layers.Embedding(input_dim = self.vocab_size,
                                                       output_dim = self.embedding_matrix.shape[-1],
                                                       input_length = input_length,
                                                       trainable = False,
                                                       embeddings_initializer = keras.initializers.Constant(self.embedding_matrix))
      )
      # 1st dropout
      trend_detection_model.add(keras.layers.Dropout(drop_rate))

      # 1st convolutional 1-D layer
      trend_detection_model.add(keras.layers.Conv1D(filters, kernel_size, padding = 'same', activation = 'relu'))

      #max pooling layer
      trend_detection_model.add(keras.layers.MaxPooling1D())

      # 2nd convolutional 1-D layer
      trend_detection_model.add(keras.layers.Conv1D(filters, kernel_size, padding = 'same', activation = 'relu'))

      #max pooling layer
      trend_detection_model.add(keras.layers.MaxPooling1D())

      # 3rd convolutional 1-D layer
      trend_detection_model.add(keras.layers.Conv1D(filters, kernel_size, padding = 'same', activation = 'relu'))

      # global max pooling layer
      trend_detection_model.add(keras.layers.GlobalAveragePooling1D())

      # 1st dense layer
      trend_detection_model.add(keras.layers.Dense(units, activation = 'relu'))

      # 2nd dropout
      trend_detection_model.add(keras.layers.Dropout(drop_rate))

      # 3rd dense layer
      trend_detection_model.add(keras.layers.Dense(units//2, activation = 'relu'))

      # 2nd dropout
      trend_detection_model.add(keras.layers.Dropout(drop_rate))

      # final dense layer
      trend_detection_model.add(keras.layers.Dense(output_dim, activation = 'sigmoid'))

      return trend_detection_model

  def train_model(self, model, X_train, y_train, X_validation=None, y_validation=None, batch_size=128, epochs=100, learning_rate=0.01):

    model.compile(loss = keras.losses.BinaryCrossentropy(),
                  optimizer = tensorflow.optimizers.Adam(learning_rate=learning_rate),
                  metrics = get_classification_metrics())

    print(model.summary())

    if not X_validation is None:
      validation_data = [X_validation, y_validation]
    else:
      validation_data = None

    early_stopping = get_earlystoping(metric='val_Precision', patience=10, mode='max')
    cp_callback = get_callback(checkpoint_path="/content/gdrive/MyDrive/Model Checkpoints/TrendDetection.ckpt", sample_per_epoch=4088, frequency=5)
    
    if not self.class_weight is None:
        history = model.fit(
            X_train,
            y_train,
            batch_size = batch_size,
            validation_data = validation_data,
            callbacks = [early_stopping, cp_callback],
            epochs = epochs,
            class_weight = self.class_weight,
            shuffle = True)
    else:
        history = model.fit(
            X_train,
            y_train,
            batch_size = batch_size,
            validation_data = validation_data,
            callbacks = [early_stopping, cp_callback],
            epochs = epochs,
            shuffle = True)

    return history

  def plot_history(self, history):
    print('1) Plot learning process based on different metrics...')
    plot_metrics(history)
    print('2) Plot learning curve...')
    return plot_history(history)

  def Evaluate_model(self, model, X_test_sequence, y_test, batch_size=128, threshold=0.3, title='', plot=True):
    prediction_prob = model.predict(X_test_sequence, verbose=1, batch_size=batch_size)
    prediction = np.where(prediction_prob >= threshold, 1, 0)
    plot_confusion_matrix(confusion_matrix(y_test, prediction), title=title, cmap ='Greens')
    print(classification_report(y_test, prediction))
    if plot:
      print('1) Plot ROC Curve...')
      plot_roc_curve(y_test, prediction, title='ROC Curve Of Model'+title, file_name=None)
      print('2) Plot AUC Curve...')
      plot_auc_curve(y_test, prediction, model_name='CNN-based-classifier', title='AUC Curve Of Model'+title, file_name = None)
      print('3) Plot Percision_Recall Curve......')
      plot_precision_recall_curve(y_test, prediction, title='Percision_Recall Curve Of Model'+title, file_name=None)

    return prediction_prob, prediction

  def topic_based_evaluation(self, model, X_test_sequence, y_test, topic_indices_test, threshold=0.3, batch_size=128, plot=False):
    for topic, indices in topic_indices_test.items():
      feature, label = filter_by_index(X_test_sequence, y_test, indices)
      title = "(Evaluation of model on {topic} data)".format(topic = topic)
      self.Evaluate_model(model, feature, label, batch_size=batch_size, threshold=threshold, title=title, plot=plot)

  def save_model(self, model, path):
    return model.save(path)

  def load_model(self, path):
    return keras.models.load_model(path)