from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sn
from utils import *
import numpy as np
import pickle
sn.set()


class Entity_Clustering():
    def __init__(self,
               clustering_model,
               list_result=[],
               ):
        self.list_result = list_result
        self.clustering_model = clustering_model


    def __feature_generator(self, embedding_matrix, occurrence_matrix, alpha=0.5):
      if (not embedding_matrix is None) and (not occurrence_matrix is None):
          return alpha * embedding_matrix + (1-alpha) * occurrence_matrix

      elif (not embedding_matrix is None):
          return embedding_matrix

      elif (not occurrence_matrix is None):
          return occurrence_matrix
      
      else:
        raise Exception('Either embedding_matrix or occurrence_matrix should be given.')

    def __clustering(self, features):
      return self.clustering_model.fit_predict(features)

    def generate_clusters(self,
                          array_entities,
                          date,
                          result_dict_path,
                          embedding_matrix=None,
                          occurrence_matrix=None,
                          similarity_saving_path = '/content/Results/Clusters_Similarity/similarity.txt'):

      feature = self.__feature_generator(embedding_matrix, occurrence_matrix)
      cluster_labels = self.__clustering(feature)

      if min(cluster_labels) == -1:
        exist_noise = True
      else:
        exist_noise = False

      print(f'==== {min(cluster_labels)} / {max(cluster_labels)}')
      cluster_labels += abs(min(cluster_labels))

      if len(self.list_result) != 0:
        last_max = max(self.list_result[-1].keys())[0]
        cluster_labels += (last_max + 1)

      try:
        with open(similarity_saving_path, 'a') as f:
          silhouette = silhouette_score(feature, cluster_labels)
          calinski_score = calinski_harabasz_score(feature, cluster_labels)
          davies_score = davies_bouldin_score(feature, cluster_labels)
          f.write(f"Clustering score for time frame {date}:\n")
          f.write(f"Silhouette Score : {silhouette.round(4)} / Calinski Harabasz : {calinski_score.round(4)} / Davies Bouldin Score : {davies_score.round(4)}\n")

      except:
        similarity_folder = similarity_saving_path[:similarity_saving_path.rfind('/')+1]
        Path(similarity_folder).mkdir(parents=True, exist_ok=True)

        with open(similarity_saving_path, 'a') as f:
          silhouette = silhouette_score(feature, cluster_labels)
          calinski_score = calinski_harabasz_score(feature, cluster_labels)
          davies_score = davies_bouldin_score(feature, cluster_labels)
          f.write(f"Clustering score for time frame {date}:\n")
          f.write(f"Silhouette Score : {silhouette.round(4)} / Calinski Harabasz : {calinski_score.round(4)} / Davies Bouldin Score : {davies_score.round(4)}\n")

      if exist_noise:
        start = cluster_labels.min()+1
      else:
        start = cluster_labels.min()

      dict_result = {}
      for i in range(start, cluster_labels.max()+1):
        values = array_entities[np.where(cluster_labels == i)[0]]
        dict_result[(i, date)] = values

      self.list_result.append(dict_result)
      with open(result_dict_path, 'wb') as handle:
        pickle.dump(self.list_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

      return cluster_labels

    def cluster2chain(self, clusters, threshold=5):
      history = []
      result = []
      for i in range(len(clusters)-1):
        common_dict = {k: v for k, v in find_common_entities(clusters[i], clusters[i+1]).items() if v >= threshold}
        if len(common_dict) >= 1:
          history.append(common_dict)
          max_bipartite = max_bipartite_matching(common_dict)
          for tup in max_bipartite:
            find_add_pair(result, tup)
            clusters[i+1][max(tup)] = np.union1d(clusters[i][min(tup)], clusters[i+1][max(tup)])
      return history, result, clusters

    def visualize_NameEntities(self, embedding, occurrence, embedding_sim, occurrence_sim, clusters, perplexity=30):
      tsne = TSNE(n_components=2, perplexity=perplexity)
      tsne_occurrence = tsne.fit_transform(occurrence)
      tsne_embedding = tsne.fit_transform(embedding)
      tsne_concat = tsne.fit_transform(np.concatenate((occurrence, embedding), axis=1))
      tsne_occurrence_sim = tsne.fit_transform(occurrence_sim)
      tsne_embedding_sim = tsne.fit_transform(embedding_sim)
      tsne_sim_concat = tsne.fit_transform(np.concatenate((occurrence_sim, embedding_sim), axis=1))

      print('----------------------------------Occurrence Matrix Only----------------------------------')
      print()
      plt.figure(figsize=(8,8))
      plt.title('Occurrence', fontsize=10)
      sn.scatterplot(x=tsne_occurrence[:,0], y=tsne_occurrence[:,-1], hue=clusters, palette='bright', legend=False)
      plt.show()

      print()
      print('----------------------------------Embedding Matrix Only----------------------------------')
      print()
      plt.figure(figsize=(8,8))
      plt.title('Embedding', fontsize=10)
      sn.scatterplot(x = tsne_embedding[:,0], y = tsne_embedding[:,-1], hue=clusters, palette='bright', legend=False)
      plt.show()

      print()
      print('----------------------------------Concatenation of Occurrence & Embedding Matrices----------------------------------')
      print()
      plt.figure(figsize=(8,8))
      plt.title('Concat(Occurrence, Embedding)', fontsize=10)
      sn.scatterplot(x = tsne_concat[:,0], y = tsne_concat[:,-1], hue=clusters, palette='bright', legend=False)
      plt.show()

      print()
      print('----------------------------------Occurrence Similarity Matrix Only----------------------------------')
      print()
      plt.figure(figsize=(8,8))
      plt.title('Occurrence Similaity', fontsize=10)
      sn.scatterplot(x = tsne_occurrence_sim[:,0], y = tsne_occurrence_sim[:,-1], hue=clusters, palette='bright', legend=False)
      plt.show()

      print()
      print('----------------------------------Embedding Similarity Matrix Only----------------------------------')
      print()
      plt.figure(figsize=(8,8))
      plt.title('Embedding Similaity', fontsize=10)
      sn.scatterplot(x = tsne_embedding_sim[:,0], y = tsne_embedding_sim[:,-1], hue=clusters, palette='bright', legend=False)
      plt.show()

      print()
      print('----------------------------------Concatenation of Similarity Matrices----------------------------------')
      print()
      plt.figure(figsize=(8,8))
      plt.title('concat(Similarities)', fontsize=10)
      sn.scatterplot(x = tsne_sim_concat[:,0], y = tsne_sim_concat[:,-1], hue=clusters, palette='bright', legend=False)
      plt.show()

      return