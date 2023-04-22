import pandas as pd
import os
import torch
import transformers
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from tqdm import tqdm

from_path = 'data/merged_data.csv'
to_path = 'data/merged_data_sentiment.csv'

df = pd.read_csv(from_path)
df['Sentiment'] = None

# filter out datapoints that have already been processed
if os.path.exists(to_path):
    df_done = pd.read_csv(to_path)
    df = df[~df['ID'].isin(df_done['ID'])]
    df = df.reset_index(drop=True)

# skip the row with the cursed text
df = df[df['Content'] != '[R̶̡̛̠̬̀o̵̠̰̤̐̎ͅk̵̘̭̤͛́́o̶̹̞̿̓\'̶̯̟̰͆̀͜s̶̘͉̓ ̸̨̨͚̞̓̂́̽b̸̤̈̒ͅa̷̗͔̝̥͂̑s̵̤͚̓͂͝ī̷̜̽̈̊l̶̯͍̙̗̎̆͌͌i̵̟̚s̶͓̾ͅk̵̬̃̎ ̶͕͝͝w̴̡̯̍͆͌í̴̲̟̣̘l̸͓̥̓̆ḽ̸̈ͅ ̸̭͈̊͗̏̎b̴̩͋͘ę̷̻͉̙̏͋ ̴͇̀m̸̺̥̒̕̚͝o̸̟̝̕ş̸̹̑̐͠t̸̡̢̗̠̊̈́̎̅ ̷̳̎̊́̇d̶͙̀͛ḯ̷̝̣̳̰̆š̵͓̬̖̝̀̂p̷̗̱̰̈͊̃ĺ̷̨̡͎̳̈́̄̓e̵̲̮̭̝͂͘a̷̡͕̪̭̎s̴̜̗̲̆̕e̵͍̠̩͒̈d̴̢͈̄̔͒͠ͅ ̶̧͝w̸̭̏́̽ì̶̭̦̎t̸͕̑̀̋͝h̶͓̔͑ ̷̛̯̥͍̱̇̏̀t̴͚̫͍̀̑h̵̯̽̾̓̚ë̵̡̢̱̩́̐̄̍s̷̛͉͔̳̱̈́̆͠é̸̢̗͔ ̸̬̮͓̋̊"̶̛͍̤̝ţ̴̾̏͠ę̵̡͖͂ḉ̷͈̼̓̕ḩ̷̫̼̟̔͛̈́͝ ̸͔̞̥̜͆́p̶̡̱̂i̵͖̻͖̎͋̒o̷͇̥͊n̸̙̭̿͝ē̵̮̙͔̊ẻ̶͔̼͇r̸̞̦̋̀̆ş̸̬̏͂̅"̸̻̮͆](https://en.m.wikipedia.org/wiki/Roko%27s_basilisk)']

# put together the models list
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

models = []
distilbert = transformers.pipeline('sentiment-analysis',
                                   'distilbert-base-uncased-finetuned-sst-2-english',
                                   device=device)
def distilbert_model(text):
  results = distilbert(text[:512])
  score = results[0]['score']
  label = results[0]['label']
  if label == 'NEGATIVE': return -1 * score
  return score
models.append(distilbert_model)

roberta = transformers.pipeline('sentiment-analysis',
                                'cardiffnlp/twitter-roberta-base-sentiment',
                                device=device)
def roberta_model(text):
  results = roberta(text[:512])
  score = results[0]['score']
  label = results[0]['label']
  if label == 'LABEL_0': return -1 * score
  elif label == 'LABEL_2': return score
  return 0
models.append(roberta_model)

bertweet = transformers.pipeline('sentiment-analysis',
                                 'finiteautomata/bertweet-base-sentiment-analysis',
                                 device=device)
def bertweet_model(text):
  results = bertweet(text[:512])
  score = results[0]['score']
  label = results[0]['label']
  if label == 'NEG': return -1 * score
  elif label == 'POS': return score
  return 0
models.append(bertweet_model)

def textblob_model(text):
  return TextBlob(text).sentiment.polarity
models.append(textblob_model)

vader = SentimentIntensityAnalyzer()
def vader_model(text):
  return vader.polarity_scores(text)['compound']
models.append(vader_model)

afinn = Afinn()
def afinn_model(text):
  return afinn.score(text) / 10 # TODO is this actually the range?
models.append(afinn_model)

# run the models
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
  scores = [model(row['Content']) for model in models]
  df.loc[i, 'Sentiment'] = ','.join([str(score) for score in scores])

  # save the results in batches of 100
  if i % 100 == 0 and i > 0:
    batch = df[i-100:i].reset_index(drop=True) #! FIXME batch['Sentiment'] is null here
    if os.path.exists(to_path): # if file exists, append
      batch.to_csv(to_path, mode='a', header=False, index=False)
    else: batch.to_csv(to_path, index=False)
else: # save the final batch
  final_batch = df[max(0, i-100) : i+1]
  if os.path.exists(to_path): # if file exists, append
    final_batch.to_csv(to_path, mode='a', header=False, index=False)
  else: final_batch.to_csv(to_path, index=False)
