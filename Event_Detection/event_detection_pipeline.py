from utils import *
import pandas as pd
import numpy as np
import pickle
import time
import glob

class Event_Detection_Pipeline():
    def __init__(self,
                 object_entity_extraction,
                 object_entity_clustering,
                 start_date,
                 end_date,
                 result_storage_path='/content/Results',
                 window_size='1D',
                 window_length=1000,
                 n_common_entity_threshold=5,
                 NE_occurrence_dict_saving_path = 'AllNameEntities.pkl',
                 clusters_dict_saving_path = 'AllClusters.pkl',
                 segmented_dfs_directory = None,
                 segmented_dfs_saving_path = 'Segmented_Dfs',
                 name_entity_loading_path= None,
                 filter_trends=False,
                 trending_threshold=0.3,
                 visualize=False,
                 ):
      
      Path(result_storage_path).mkdir(parents=True, exist_ok=True)
      end_of_windows = pd.date_range(start=start_date, end=end_date, freq=window_size, tz='UTC')
      if segmented_dfs_directory is None:
        dfs_segmented_data = object_entity_extraction.segment_dataframe(start_date,
                                                                       end_date,
                                                                       window_size,
                                                                       window_length)
        store_dfs(dfs_segmented_data,
                  end_of_windows,
                  path=result_storage_path + '/' + segmented_dfs_saving_path)
      else:
        dfs_segmented_data = []
        for filepath in sorted(glob.iglob(segmented_dfs_directory + '/*.csv')):
          dfs_segmented_data.append(pd.read_csv(filepath))

      if filter_trends:
        trending_labels = np.empty(shape=(len(end_of_windows), len(dfs_segmented_data[0])), dtype=bool)

      if not name_entity_loading_path is None:
        list_entities = format_entities(name_entity_loading_path)

      for i, df in enumerate(dfs_segmented_data):
        start_time = time.time()

        if filter_trends:
          print('0. Trending Data Extraction...')
          df, trending_labels[i] = object_entity_extraction.trending_data_extractor.predict(df=df, threshold=trending_threshold)

        print(f'---------------------------------------- Working on window {i+1} with {len(df)} samples')

        if not name_entity_loading_path is None:
          print('1. Restoring name entities...')
          entities = list_entities[i]
        else:
          print('1. Extract existing name entities in time frame...')
          entities = object_entity_extraction.text2entity(df)

        print('2. Calculate co-occurrence matrix...')
        entities, occurrence_matrix = object_entity_extraction.generate_occurrence_matrix(df,
                                                                                         entities,
                                                                                         NE_occurrence_dict_saving_path=result_storage_path + '/' + NE_occurrence_dict_saving_path,
                                                                                         date=end_of_windows[i].strftime('%Y-%m-%d / %H:%M:%S'))

        occurrence_distance_matrix = object_entity_extraction.generate_distance_matrix(occurrence_matrix)

        print('3. Calculate embedding matrix...')
        embedding_matrix = object_entity_extraction.generate_embedding_matrix(df,
                                                                              occurrence_matrix,
                                                                              entities,
                                                                              context_selector=['retweet_count', 4])

        embedding_distance_matrix = object_entity_extraction.generate_distance_matrix(embedding_matrix)

        print('4. Cluster name entities...')
        lables = object_entity_clustering.generate_clusters(array_entities=np.array(list(entities), dtype=str),
                                                            date=end_of_windows[i].strftime('%Y-%m-%d / %H:%M:%S'),
                                                            result_dict_path=result_storage_path + '/' + clusters_dict_saving_path,
                                                            embedding_matrix=embedding_distance_matrix,
                                                            occurrence_matrix = occurrence_distance_matrix)

        print(f'---------------------------------------- Elapsed Time : {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

        if visualize:
          print('4.1 Visualizing the clusters...')
          object_entity_clustering.visualize_NameEntities(embedding=embedding_matrix,
                                                          occurrence=occurrence_matrix,
                                                          embedding_sim=embedding_distance_matrix,
                                                          occurrence_sim=occurrence_distance_matrix,
                                                          clusters=lables)

      print(f'-------------------------------------------------------------------------------------------------')
      print('5. Create Cluster Chain...')
      self.history, self.final_result, self.incremental_clusters = object_entity_clustering.cluster2chain(object_entity_clustering.list_result,
                                                                                                         threshold=n_common_entity_threshold)
      with open(result_storage_path + '/' + clusters_dict_saving_path, 'rb') as f:
        self.list_clusters = pickle.load(f)
      
      all_NE_paths = name_entity_loading_path if not name_entity_loading_path is None else result_storage_path + '/' + NE_occurrence_dict_saving_path
      segmented_dfs_path = segmented_dfs_directory if not segmented_dfs_directory is None else result_storage_path + '/' + segmented_dfs_saving_path

      print('6. Saving the results...')
      self.__save_results(result_storage_path,
                          all_NE_paths,
                          segmented_dfs_path,
                          start_date,
                          end_date,
                          window_size)

    def __save_results(self,
                       results_path,
                       all_NE_paths,
                       segmented_dfs_path,
                       start_date,
                       end_date,
                       window_size):
      
      
      end_of_windows = pd.date_range(start=start_date, end=end_date, freq=window_size, tz='UTC').strftime('%Y-%m-%d / %H:%M:%S')
      for chain_id, cluster_chain in enumerate(self.final_result):
        save_event_detail(cluster_chain,
                          self.list_clusters,
                          chain_id,
                          results_path,
                          end_of_windows)

      for filepath in sorted(glob.iglob(results_path + '/event*')):
        tweet_hashtag_retrieve(event_path=filepath,
                               NE_paths=all_NE_paths,
                               df_paths=segmented_dfs_path,
                               end_of_windows=end_of_windows,
                               export_path=filepath)
      return