"""
Python ETL
"""

import requests
import pandas as pd
from sqlalchemy import create_engine


# This API extracts data from: http://universities.hipolabs.com
def extract()-> dict:
    API_URL = "http://universities.hipolabs.com/search?country=United+States"

    data = requests.get(API_URL).json()

    return data

# Transforms the dataset into desired structure and filters
def transform(data:dict) -> pd.DataFrame:
    df = pd.DataFrame(data)

    print(f"Total Number of universities from API {len(data)}")

    df = df[df["name"].str.contains("California")]

    print(f"Number of universities in california {len(df)}")

    df['domains'] = [','.join(map(str, l)) for l in df['domains']]
    df['web_pages'] = [','.join(map(str, l)) for l in df['web_pages']]

    df = df.reset_index(drop=True)

    return df[["domains","country","web_pages","name"]]


# Loads data into a sqllite database
def load(df:pd.DataFrame)-> None:
    disk_engine = create_engine('sqlite:///my_lite_store.db')
    
    df.to_sql('cal_uni', disk_engine, if_exists='replace')
