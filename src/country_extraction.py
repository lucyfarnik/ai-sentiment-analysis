from functools import partial
import time
import pandas as pd
import pycountry
from geopy.geocoders import Nominatim

# file_path = 'data/merged_data.csv'
file_path = 'data/merged_data_sentiment.csv'

df = pd.read_csv(file_path)
df['Country'] = None

geolocator = Nominatim(user_agent="ads-coursework")
for i, row in df.iterrows():
  if i % 100 == 0:
    print(f"{i=}")

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

    # add some delays to avoid overloading the geopy API
    if i % 1000 == 0: time.sleep(10)
    elif i % 100 == 0: time.sleep(1)
    continue

  if len(location) == 2:
    country_obj = pycountry.countries.get(alpha_2=location)
  else:
    country_obj = pycountry.countries.get(alpha_3=location)
  df.at[i, 'Country'] = country_obj.alpha_3

df.to_csv(file_path, index=False)
