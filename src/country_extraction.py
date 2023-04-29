from functools import partial
import time
import os
import pandas as pd
import pycountry
from geopy.geocoders import Nominatim

def extract_country():
  file_path = 'data/merged_data_sentiment.csv'
  temp_file_path = 'data/merged_data_sentiment_temp.csv'

  df = pd.read_csv(file_path)
  if os.path.exists(temp_file_path):
    df_done = pd.read_csv(temp_file_path)
    ids_done = df_done['ID'].values
  else:
    ids_done = []
  if 'Country' not in df.columns:
    df['Country'] = None

  geolocator = Nominatim(user_agent="ads_coursework")
  for i, row in df.iterrows():
    # if the country has already been extracted, just add the country code already there and skip
    if row['ID'] in ids_done:
      df.at[i, 'Country'] = df_done[df_done['ID'] == row['ID']]['Country'].values[0]
      continue

    if i % 100 == 0:
      print(f"{i=}")
    if (i > 64000 and i % 10 == 0) or (i % 1000 == 0 and i > 0):
      # save temporary progress (only the part already processed)
      df.iloc[:i].to_csv(temp_file_path, index=False)

    location = row['Location']
    if type(location) is not str:
      continue
    
    if len(location) > 3:
      # preprocessing to extract the country
      geocode = partial(geolocator.geocode, language="en")
      address = geocode(location)
      if address is not None:
        address_str = address.address
        address_list = address_str.split(',')
        country = address_list[-1].strip()
      else:
        # handle edge cases - geopy sometimes can't parse certain addresses
        if location[-5:] in ['India', '𝘐𝘯𝘥𝘪𝘢']:
          country = 'India'
        elif location[-7:] == 'Estonia':
          country = 'Estonia'
        elif location[-5:] in ['Maine', 'Miami']:
          country = 'United States'
        elif location[-7:] == 'Chicago':
          country = 'United States'
        elif location[-9:] == 'Australia':
          country = 'Australia'
        else:
          print(f"Location couldn't be parsed: {location=}")
          df.at[i, 'Country'] = None
          continue
      country_obj = pycountry.countries.search_fuzzy(country)
      if len(country_obj) == 0:
        print(f"Location couldn't be parsed: {country=} (location = {location})")
        df.at[i, 'Country'] = None
        continue
      df.at[i, 'Country'] = country_obj[0].alpha_3

      # add delays to avoid overloading the geopy API and violating their user agreement
      # if row['Platform'] == 'Twitter':
      #   timeout_multiplier = max(1, (i-65000)/1000)
      #   if i % 100 == 0: time.sleep(10 * timeout_multiplier)
      #   elif i % 10 == 0: time.sleep(timeout_multiplier)
      # else:
      #   if i % 1000 == 0: time.sleep(10)
      #   elif i % 100 == 0: time.sleep(1)
      time.sleep(1)
      continue

    if len(location) == 2:
      country_obj = pycountry.countries.get(alpha_2=location)
    else:
      country_obj = pycountry.countries.get(alpha_3=location)
    df.at[i, 'Country'] = country_obj.alpha_3

  df.to_csv(file_path, index=False)

  # delete temporary file
  if os.path.exists(temp_file_path):
    os.remove(temp_file_path)

if __name__ == '__main__':
  extract_country()
