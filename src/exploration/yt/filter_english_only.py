import pandas as pd

df = pd.read_csv('data/yt_w_langs.csv')
df = df[df['lang'] == 'en']
df.to_csv('data/yt_w_langs_en.csv', index=False)
