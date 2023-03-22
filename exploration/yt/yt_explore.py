import requests
import transformers
import numpy as np

key = 'AIzaSyDvuHBARz2o65ylSX_orvHASPCN0xHtxes'
classifier = transformers.pipeline("sentiment-analysis")

vid_res = requests.get('https://www.googleapis.com/youtube/v3/search', {
    'key': key,
    'part': 'snippet',
    'q': 'AI',
    'publishedBefore': '2018-01-01T00:00:00Z',
}).json()
videos = [{'id': x['id']['videoId'], 'title': x['snippet']['title']}
           for x in vid_res['items']]

for vid in videos:
  com_res = requests.get('https://www.googleapis.com/youtube/v3/commentThreads', {
      'key': key,
      'part': 'snippet',
      'videoId': vid['id'],
      'maxResults': 100,
  }).json()
  if 'error' in com_res: continue
  comments = [{'id': x['id'],
              'text': x['snippet']['topLevelComment']['snippet']['textDisplay'],
              'channel': x['snippet']['topLevelComment']['snippet']['authorChannelId']['value'],
              }
              for x in com_res['items']]

  for com in comments:
    cls_res = classifier(com['text'][:512])[0]
    cls_results = [[x['text'], ] for x in comments]
    print(f"\"{com['text']}\" is {cls_res['label']} ({100*cls_res['score']:.1f}%)\n")

  chan_res = requests.get('https://www.googleapis.com/youtube/v3/channels', {
      'key': key,
      'part': 'snippet',
      'id': ','.join([c['channel'] for c in comments[:50]]),
      'maxResults': 50,
  }).json()
  has_country = np.array(['country' in c['snippet'] for c in chan_res['items']])
  print(f"{100*has_country.sum()/has_country.size:.2f}% have country codes")
