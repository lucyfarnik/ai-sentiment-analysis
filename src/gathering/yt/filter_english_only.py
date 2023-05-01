import pandas as pd
import os

# get all files (w_langs)
dfs = []
for file in os.listdir('data/yt_stages/w_langs'):
  dfs.append(pd.read_csv('data/yt_stages/w_langs/' + file))
df = pd.concat(dfs)

df = df[df['lang'] == 'en']
df.to_csv('data/youtube_data.csv', index=False)

print(f"English only: {df.shape[0]} rows")
