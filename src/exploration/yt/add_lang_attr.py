import pandas as pd
import transformers
import math
from tqdm import tqdm
import os

df = pd.read_csv('data/yt_data.csv')

# check which rows have already been processed
df_done = pd.read_csv('data/yt_w_langs.csv')
done_count = df_done.shape[0]

lang_cls = transformers.pipeline('text-classification',
                                 'papluca/xlm-roberta-base-language-detection',
                                 device='mps')
batch_size = 20
file_path = 'data/yt_w_langs.csv'
for batch_i in tqdm(range(done_count//batch_size, math.ceil(df.shape[0] / batch_size))):
  batch = df[batch_size*batch_i : batch_size*(batch_i+1)] # slice data into batches
  batch_text = batch['text'].to_list()
  for i, txt in enumerate(batch_text):
    if type(txt) is not str: # some strings may be null, nan etc
      batch_text[i] = ''
    elif len(txt) > 400: # some strings may be too long (doesn't effect language classification)
      batch_text[i] = txt[:400]
  langs = lang_cls(batch_text) # classify
  for i, lang in enumerate(langs): # add into dataframe
    df.loc[batch_size*batch_i + i, 'lang'] = lang['label']
  batch = df[batch_size*batch_i : batch_size*(batch_i+1)]
  # save
  if batch_i > 0 or os.path.exists(file_path): # if file exists, append
    batch.to_csv(file_path, mode='a', header=False, index=False)
  else: batch.to_csv(file_path, index=False)
