import pandas as pd
import torch
import transformers
import math
import os
from tqdm import tqdm
import json

file_path = 'data/proccessed/yt_w_sentiment.csv'
batch_size = 20

df = pd.read_csv('data/yt_w_langs_en.csv')
df['sentiment'] = None

# check which rows have already been processed
if os.path.exists(file_path):
  df_done = pd.read_csv(file_path)
  done_count = df_done.shape[0]
else: done_count = 0

if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

# Load models
checkpoints = [
   'distilbert-base-uncased-finetuned-sst-2-english',
   'cardiffnlp/twitter-roberta-base-sentiment',
   'finiteautomata/bertweet-base-sentiment-analysis',
   'j-hartmann/emotion-english-distilroberta-base',
]
models = [transformers.pipeline('sentiment-analysis', checkpoint, device=device)
          for checkpoint in checkpoints]

def classify(inputs):
    model_results = [model(inputs) for model in models]
    results = [[] for _ in range(len(inputs))]
    for i, _ in enumerate(inputs):
        for model_result in model_results:
            results[i].append(model_result[i])
    return results

for batch_i in tqdm(range(done_count//batch_size, math.ceil(df.shape[0] / batch_size))):
    batch = df[batch_size*batch_i : batch_size*(batch_i+1)] # slice data into batches
    batch_text = batch['text'].to_list()
    for i, txt in enumerate(batch_text):
        if type(txt) is not str: # some strings may be null, nan etc
            batch_text[i] = ''
        elif len(txt) > 400: # some strings may be too long (doesn't effect language classification)
            batch_text[i] = txt[:400]
    sentiment = classify(batch_text) # classify
    for i, sent in enumerate(sentiment): # add into dataframe
        df.loc[batch_size*batch_i + i, 'sentiment'] = json.dumps(sent)
    batch = df[batch_size*batch_i : batch_size*(batch_i+1)]
    # save
    if batch_i > 0 or os.path.exists(file_path): # if file exists, append
        batch.to_csv(file_path, mode='a', header=False, index=False)
    else: batch.to_csv(file_path, index=False) 