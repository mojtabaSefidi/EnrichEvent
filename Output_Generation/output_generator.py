from Event_Summarization.predict import Summary_Generator
from wordcloud_fa import WordCloudFa
from PIL import Image
from utils import *
import pandas as pd
import numpy as np
import itertools
import random
import glob
import json
import ast



class Output_Generator():
  def __init__(self,
               summarization_model_path,
               embedding_model_path,
               gpt_model_path):
      
      self.summarzation_genarator = Summary_Generator(summarization_model_path=summarization_model_path,
                                                      embedding_model_path=embedding_model_path,
                                                      gpt_model_path=gpt_model_path)


  def __grey_color_func(self):
      return "hsl(203, 100%%, %50d%%)" % random.randint(60, 100)

  def __generate_wordcloud(self,
                          event_path,
                          font_address,
                          background_image_address,
                          max_font_size=120,
                          show_result=False):
    df = pd.read_csv(event_path + '/final_df.csv')
    result = ''
    for text, hashtags in df[['text', 'hashtags']].itertuples(index=False):
      text = text.replace('ج .  ا', 'جمهوری اسلامی ایران')
      hashtags = ast.literal_eval(str(hashtags))
      if type(hashtags) == list:
        text = remove_useless_hashtags(text, hashtags)
      result = result + text + ' '
    result = remove_stopwords(result.strip())

    wordcloud = WordCloudFa(font_path=font_address,
                            mask=np.array(Image.open(background_image_address)),
                            no_reshape=True,
                            include_numbers=False,
                            max_font_size=max_font_size)
                            # color_func=self.__grey_color_func)

    word_cloud = wordcloud.generate(result)
    image = word_cloud.to_image()
    if show_result:
      image.show()
    event_id = event_path[event_path.rfind('_')+1:]
    export_path = event_path + '/WordCloud_event_' + event_id + '.png'
    image.save(export_path)
    return

  def __hashtags_detail(self, event_path, k=5):
    with open(event_path + '/hashtags_all.txt', 'r', encoding='utf8') as f:
      contents = f.read().splitlines()

    num_unique = int(contents[0].split(' ')[0])
    del contents[0]

    contents = [content.split(':') for content in contents]
    if contents[-1]==['']:
      del contents[-1]
    most_frequnt = {}
    if k > len(contents):
      k = len(contents)
    for i in range(k):
      most_frequnt[contents[i][0].strip()] = int(contents[i][-1].strip())
    sum = 0
    for content in contents:
      sum += int(content[-1].strip())
    return num_unique, sum, most_frequnt

  def __frequent_users(self, event_path, k=5):
    with open(event_path + '/users_freq_all.txt', 'r', encoding='utf8') as f:
      contents = f.read().splitlines()
    contents = [content.split(':') for content in contents]
    most_frequnt = {}
    if k > len(contents):
      k = len(contents)
    for i in range(k):
      most_frequnt[contents[i][0].strip()] = int(contents[i][-1].strip())
    return dict(sorted(most_frequnt.items(), key=lambda item: item[1], reverse=True))

  def __NE_details(self, event_path, k=5):
    unique_entities = []
    for path in sorted(glob.iglob(event_path + '/window*')):
      with open(path, 'r', encoding='utf8') as f:
        contents = f.read().splitlines()
      unique_entities = np.union1d(unique_entities, contents)
    unique_entities
    result = {entity:0 for entity in unique_entities}
    df = pd.read_csv(event_path + '/final_df.csv')
    for text in df['text']:
      tokens = text.split()
      for entity in unique_entities:
        result[entity] = result[entity] + tokens.count(entity)

    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return len(result), sum(result.values()), dict(itertools.islice(result.items(), k))

  def __user_diversity(self, event_path):
    entropy = []
    scale = lambda x: -1 * x * np.log2(x)
    with open(event_path + '/users_freq_time_frame.txt', 'r', encoding='utf8') as f:
      contents = f.read().splitlines()
    result = []
    d = {}
    i = 1
    while i < len(contents):
      if contents[i] != '':
        temp = contents[i].split(':')
        freq = d.get(temp[0].strip(), -1)
        if freq == -1:
          d[temp[0].strip()] = int(temp[-1])
        else:
          d[temp[0].strip()] = int(temp[-1]) + freq
        i+=1
      else:
        result.append(d)
        d = {}
        i+=2
    for dictionary in result:
      temp = np.array(list(dictionary.values()), int)/sum(dictionary.values())
      entropy.append(scale(temp).sum())
    return round(np.mean(entropy), 4)

  def __extract_attributs(self, event_path, summarization, export_path=''):
    df = pd.read_csv(event_path + '/final_df.csv')
    output = {}
    output['Event ID'] = event_path[event_path.rfind('_')+1:]
    output['Event Date Range'] = 'Since ' + df['datetime'][0] + ' Until ' + df['datetime'][len(df)-1]
    output['Event Summarization'] = summarization

    most_like, num_like = df.sort_values(['like_count'], ascending=False)['text'].to_numpy()[0], df.sort_values(['like_count'], ascending=False)['like_count'].to_numpy()[0]
    most_retweet, num_retweet = df.sort_values(['retweet_count'], ascending=False)['text'].to_numpy()[0], df.sort_values(['retweet_count'], ascending=False)['retweet_count'].to_numpy()[0]
    most_reply, num_reply = df.sort_values(['reply_count'], ascending=False)['text'].to_numpy()[0], df.sort_values(['reply_count'], ascending=False)['reply_count'].to_numpy()[0]

    if num_like != 0:
      output['Tweet with the highest like count'] = {most_like : num_like}
    else:
      output['Tweet with the highest like count'] = {'No tweet has more than 0 likes !!'}

    if num_retweet != 0:
      output['Tweet with the highest retweet count'] = {most_retweet : num_retweet}
    else:
      output['Tweet with the highest retweet count'] = {'No tweet has more than 0 retweet !!'}

    if num_reply != 0:
      output['Tweet with the highest reply count'] = {most_reply : num_reply}
    else:
      output['Tweet with the highest reply count'] = {'No tweet has more than 0 reply !!'}


    output['Number of unique hashtags'] ,output['Number of all the posted hashtags'], output['Most Frequent hashtags'] = self.__hashtags_detail(event_path)

    output['Number of unique users'] = df['username'].nunique()
    output['Average User Diversity'] = self.__user_diversity(event_path)
    output['Most Frequent users'] = self.__frequent_users(event_path)

    output['Number of unique named entities'], output['Number of all the appeared name entities'], output['Most Frequent named entities'] = self.__NE_details(event_path)
    with open(export_path + '/detail_event_' + event_path[event_path.rfind('_')+1:] + '.json', 'w', encoding='utf8') as f:
      json.dump(output, f, default=str, ensure_ascii=False, indent=4)
    return

  def export(self,
             results_path,
             eps=0.4,
             min_samples=1,
             num_clusters=3,
             number_prediction=5,
             min_length=30,
             max_length=100,
             context_length=3,
             font_address='Essential_Files/Lalezar-Regular.ttf',
             background_image_address='Essential_Files/Twitter.png'):

    for event_path in sorted(glob.iglob(results_path + '/event*')):
      event_id = event_path[event_path.rfind('_')+1:]
      print(f'-------------------------- Working on event {event_id} --------------------------')

      print('1) Drawing WordCloud...')
      self.__generate_wordcloud(event_path=event_path,
                                font_address=font_address,
                                background_image_address=background_image_address)

      print('2) Generating Summary...')
      _, summary = self.summarzation_genarator.generate_event_summary(event_path=event_path,
                                                                      eps=eps,
                                                                      min_samples=min_samples,
                                                                      num_clusters=num_clusters,
                                                                      number_prediction=number_prediction,
                                                                      min_length=min_length,
                                                                      max_length=max_length,
                                                                      context_length=context_length)


      print("3) Extract Events' Attributs...")
      self.__extract_attributs(event_path, summary[0], export_path=event_path)
    return
