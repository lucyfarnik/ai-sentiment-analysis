import os
import sys
import math
import requests
import transformers
import pandas as pd

sys.path.append(os.path.abspath('.'))
from src.constants import google_key as key

num_vids = 1e6
query = 'AI|"artificial intelligence"|ChatGPT|OpenAI|DeepMind -ad -free -purchase -premium -avail -claim -giveaway -participants -Telegram -winner -win -credits -token -tokens -aiART -artwork -art -cosplay -character -waifu -generated -"470EX-AI"'
date_range_start = '2013-01-01T00:00:00Z'
date_range_end = None
should_classify = False
file_path = 'data/yt_raw/yt_data3.csv'

# init the classifier
classifier = transformers.pipeline('sentiment-analysis',
                                   'distilbert-base-uncased-finetuned-sst-2-english')

next_vid_page_token = None
for vid_page_i in range(math.ceil(num_vids / 50)):
  comments = []
  # fetch some videos - construct params
  vid_req_params = {
      'key': key,
      'part': 'snippet',
      'q': query,
      'maxResults': 50 if num_vids >= 50 else num_vids,
      'relevanceLanguage': 'en',
  }
  if next_vid_page_token is not None: # pagination
    if next_vid_page_token == '_INVALID_':
      print(f"Invalid next video page token for {vid_page_i=}")
      break
    vid_req_params['pageToken'] = next_vid_page_token

  # date range
  if date_range_start is not None:
    vid_req_params['publishedAfter'] = date_range_start
  if date_range_end is not None:
    vid_req_params['publishedBefore'] = date_range_end
  
  # note that doing search programmatically uses 100 quota points while the other requests only take 1
  # we could get around the quota by doing this part with web scraping
  # (we can currently get ~100k comments per day (per API key))
  vid_res = requests.get('https://www.googleapis.com/youtube/v3/search',
                         vid_req_params).json()
  if 'error' in vid_res: # TODO error handling
    print(vid_res['error'])
    continue
  # store page token
  if 'nextPageToken' in vid_res: next_vid_page_token = vid_res['nextPageToken']
  else: next_vid_page_token = '_INVALID_'
  videos = [{'id': x['id']['videoId'],
            'title': x['snippet']['title'],
            'date': x['snippet']['publishedAt'],
            }
            for x in vid_res['items']
            if 'id' in x and 'videoId' in x['id']]

  # for each video, get the comments
  for vid_i, vid in enumerate(videos):
    print(f'{vid_page_i=} {vid_i=}')
    # fetch comments (with pagination so that we get all of them)
    next_com_page_token = None
    vid_comments = []
    for com_page_i in range(100):
      if com_page_i > 0: print(f'{com_page_i=}')
      com_req_params = {
          'key': key,
          'part': 'snippet',
          'videoId': vid['id'],
          'maxResults': 100,
          'textFormat': 'plainText',
      }
      if next_com_page_token is not None: # pagination
        if next_com_page_token == '_INVALID_': break
        com_req_params['pageToken'] = next_com_page_token
      com_res = requests.get('https://www.googleapis.com/youtube/v3/commentThreads',
                             com_req_params).json()
      if 'error' in com_res: # TODO error handling
        if 'disabled comments.' not in com_res['error']['message']:
          print(com_res['error'])
        continue
      # store page token
      if 'nextPageToken' in com_res: next_com_page_token = com_res['nextPageToken']
      else: next_com_page_token = '_INVALID_'
      if len(com_res['items']) == 0:
        # no comments
        continue
      for com in com_res['items']:
        try:
          vid_comments.append({'id': com['id'],
                              'text': com['snippet']['topLevelComment']['snippet']['textDisplay'],
                              'username': com['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                              'date': com['snippet']['topLevelComment']['snippet']['publishedAt'],
                              'country': None,
                              'likes': com['snippet']['topLevelComment']['snippet']['likeCount'],
                              'n_children': com['snippet']['totalReplyCount'], # TODO: get the children
                              'title': vid['title'],
                              'platform': 'youtube',
                              'meta': {
                                'vid_id': vid['id'],
                                'user_id': com['snippet']['topLevelComment']['snippet']['authorChannelId']['value'],
                                'vid_date': vid['date'],
                              }
                              })
        except:
          pass


    if should_classify:
      # classify the text
      cls_res = classifier([com['text'][:512] for com in vid_comments])
      for com_i, cls in enumerate(cls_res):
        vid_comments[com_i]['sentiment_label'] = cls['label']
        vid_comments[com_i]['sentiment_score'] = cls['score']

    # get the channel of the users who posted each comment to find the country
    # TODO: this could be more efficient - sometimes vid_comments is short, we can instead fetch comments for the `comments` array all at once to better batch things
    for chan_page_i in range(math.ceil(len(vid_comments) / 50)):
      page_comments = vid_comments[chan_page_i*50 : (chan_page_i+1)*50]
      chan_res = requests.get('https://www.googleapis.com/youtube/v3/channels', {
          'key': key,
          'part': 'snippet',
          'id': ','.join([c['meta']['user_id'] for c in page_comments]),
          'maxResults': 50,
      }).json()
      if 'error' in chan_res: # TODO error handling
        print(chan_res['error'])
        continue
      for chan in chan_res['items']: # match the channel with the comment (the order isn't synced)
        if 'country' not in chan['snippet']: continue
        for com in vid_comments:
          if com['meta']['user_id'] == chan['id']:
            com['country'] = chan['snippet']['country']
    
    # append to global results array
    for com in vid_comments: comments.append(com)

  # save the fetched data for this page
  df = pd.DataFrame(comments)
  print(f'{df.shape=}')
  if vid_page_i > 0 or os.path.exists(file_path): # if file exists, append
    df.to_csv(file_path, mode='a', header=False, index=False)
  else: df.to_csv(file_path, index=False)
