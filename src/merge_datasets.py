import pandas as pd

def merge_datasets():
  # Read the youtube data
  youtube = pd.read_csv('data/youtube_data.csv')
  # rename from the naming scheme we originally agreed on to the new one
  youtube = youtube.rename(columns={
    'id': 'ID',
    'text': 'Content',
    'username': 'User',
    'date': 'Date',
    'country': 'Location',
    'likes': 'Reactions',
    'n_children': 'N_Children',
    'title': 'Post Title',
    'platform': 'Platform',
  })
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
  twitter = pd.read_csv('tweets2020.csv', dtype={'ID': str})
  for i, row in twitter.iterrows():
    date = row['Date']
    if len(date) > 10:
      date = date[:10]
    if '/' in date:
      month, day, year = date.split('/')
      date = f"{year}-{month}-{day}"
    twitter.at[i, 'Date'] = date
  print(f"Twitter: {twitter.shape[0]} rows")

  # Merge the datasets
  df = pd.concat([youtube, reddit, twitter])
  df = df.sort_values(by=['Date'])
  df.to_csv('data/merged_data.csv', index=False)
  print(f"Merged: {df.shape[0]} rows")

if __name__ == '__main__':
  merge_datasets()
