import pandas as pd
from os.path import exists

train_df = pd.read_csv("./data/train.csv")
train_set1_df = pd.read_csv("./data/train_set1.csv")
train_set2_df = pd.read_csv("./data/train_set2.csv")

test_df = pd.read_csv("./data/test.csv")
test_set1_df = pd.read_csv("./data/test_set1.csv")
test_set2_df = pd.read_csv("./data/test_set2.csv")

train_df.to_pickle('./data/train.pkl')
train_set1_df.to_pickle('./data/train_set1.pkl')
train_set2_df.to_pickle('./data/train_set2.pkl')

test_df.to_pickle('./data/test.pkl')
test_set1_df.to_pickle('./data/test_set1.pkl')
test_set2_df.to_pickle('./data/test_set2.pkl')