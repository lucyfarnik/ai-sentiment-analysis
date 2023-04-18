import pandas as pd

# Read the youtube data
youtube = pd.read_csv('data/youtube_data.csv')
youtube = youtube.dropna(subset=['Date'])
youtube = youtube[youtube['Date'] != 'CZ']
for i, row in youtube.iterrows():
  youtube.at[i, 'Date'] = row['Date'][:10]
print(f"Youtube: {youtube.shape[0]} rows")

# Read the reddit data
reddit = pd.read_csv('reddit_data.csv')
for i, row in reddit.iterrows():
  day, month, year = row['Date'].split(' ')[0].split('/')
  reddit.at[i, 'Date'] = f'{year}-{month}-{day}'
print(f"Reddit: {reddit.shape[0]} rows")

# Read the twitter data
twitter = pd.read_csv('tweets.csv')
for i, row in twitter.iterrows():
  twitter.at[i, 'Date'] = row['Date'][:10]
print(f"Twitter: {twitter.shape[0]} rows")

# Merge the datasets
df = pd.concat([youtube, reddit, twitter])
df = df.sort_values(by=['Date'])
df.to_csv('data/merged_data.csv', index=False)
print(f"Merged: {df.shape[0]} rows")
