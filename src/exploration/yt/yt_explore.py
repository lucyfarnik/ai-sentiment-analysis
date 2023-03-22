import os
import sys
import math
import requests
import transformers
import pandas as pd

sys.path.append(os.path.abspath('.'))
from src.constants import google_key as key

num_comments = 1e5
query = 'AI'
date_range_start = '2018-01-01T00:00:00Z'
date_range_end = '2020-01-01T00:00:00Z'
should_classify = False

# init the classifier
classifier = transformers.pipeline('sentiment-analysis',
                                   'distilbert-base-uncased-finetuned-sst-2-english')

comments = []
next_page_token = None
for page_i in range(math.ceil(num_comments / (50*50))):
  # fetch some videos - construct params
  vid_req_params = {
      'key': key,
      'part': 'snippet',
      'q': query,
      'maxResults': 50,
  }
  if next_page_token is not None: # pagination
    vid_req_params['pageToken'] = next_page_token

  # date range
  if date_range_start is not None:
    vid_req_params['publishedAfter'] = date_range_start
  if date_range_end is not None:
    vid_req_params['publishedBefore'] = date_range_end
  
  # note that doing search programmatically uses 100 quota points while the other requests only take 1
  # we could get around the quota by doing this part with web scraping
  vid_res = requests.get('https://www.googleapis.com/youtube/v3/search',
                         vid_req_params).json()
  if 'error' in vid_res: # TODO error handling
    print(vid_res['error'])
    continue
  next_page_token = vid_res['nextPageToken'] # store page token
  videos = [{'id': x['id']['videoId'],
            'title': x['snippet']['title'],
            'date': x['snippet']['publishedAt'],
            }
            for x in vid_res['items']
            if 'id' in x and 'videoId' in x['id']]

  # for each video, get the comments
  for vid_i, vid in enumerate(videos):
    print(f'{page_i=} {vid_i=}')
    # fetch comments
    com_res = requests.get('https://www.googleapis.com/youtube/v3/commentThreads', {
        'key': key,
        'part': 'snippet',
        'videoId': vid['id'],
        'maxResults': 50,
    }).json()
    if 'error' in com_res: # TODO error handling
      if 'disabled comments.' not in com_res['error']['message']:
        print(com_res['error'])
      continue
    if len(com_res['items']) == 0:
      # no comments
      continue
    vid_comments = [{'id': x['id'],
                'text': x['snippet']['topLevelComment']['snippet']['textDisplay'],
                'channel': x['snippet']['topLevelComment']['snippet']['authorChannelId']['value'],
                'date': x['snippet']['topLevelComment']['snippet']['publishedAt'],
                'vid_id': vid['id'],
                'vid_title': vid['title'],
                'vid_date': vid['date'],
                }
                for x in com_res['items']]

    if should_classify:
      # classify the text
      cls_res = classifier([com['text'][:512] for com in vid_comments])
      for com_i, cls in enumerate(cls_res):
        vid_comments[com_i]['sentiment_label'] = cls['label']
        vid_comments[com_i]['sentiment_score'] = cls['score']

    # get the channel of the user who posted it to find the country
    chan_res = requests.get('https://www.googleapis.com/youtube/v3/channels', {
        'key': key,
        'part': 'snippet',
        'id': ','.join([c['channel'] for c in vid_comments[:50]]),
        'maxResults': 50,
    }).json()
    if 'error' in chan_res: # TODO error handling
      print(chan_res['error'])
      continue
    for chan in chan_res['items']: # match the channel with the comment (the order isn't synced)
      country = chan['snippet']['country'] if 'country' in chan['snippet'] else None
      for com in vid_comments:
        if com['channel'] == chan['id']: com['country'] = country
    
    # append to global results array
    for com in vid_comments: comments.append(com)

# save the fetched data
df = pd.DataFrame(comments)
print(df.shape, df.head())
df.to_csv('data/yt_data.csv')
