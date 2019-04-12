import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description='ETL Pipeline for disaster data',
)
parser.add_argument('-messages', action='store', type=str, help = 'The messages file to be import', default = 'messages.csv')
parser.add_argument('-categories', action = 'store', type = str, help = 'The category file to be import', default = 'categories.csv')

args = parser.parse_args()

messages = pd.read_csv(vars(args)['messages'])
categories = pd.read_csv(vars(args)['categories'])
df = messages.merge(categories, on='id')
categories = df.categories.str.split(';', expand = True)
row = categories.iloc[0, :]
category_colnames = list(map((lambda x: x[:-2]), row))
categories.columns = category_colnames

for column in categories:
    # set each value to be the last character of the string
    categories[column] = list(map(lambda x: x[-1:], categories[column]))
    
    # convert column from string to numeric
    categories[column] = list(map(int, categories[column]))
categories.head()

df.drop('categories', axis = 1, inplace = True)
df = pd.concat([df, categories], axis = 1)
df.drop_duplicates(inplace = True)
print(df.head())
print("All done")
from sqlalchemy import create_engine
engine = create_engine('sqlite:///disaster.db')
df.to_sql('messages', engine, index=False)
