import pandas as pd

# split the yt_w_langs.csv file into 3 smaller files

df = pd.read_csv('data/yt_raw/yt_w_langs.csv')
parts = 3
for i in range(parts):
    df.iloc[int(i*len(df)/parts):int((i+1)*len(df)/parts)].to_csv(f'data/yt_raw/yt_w_langs/yt_w_langs_{i}.csv', index=False)
