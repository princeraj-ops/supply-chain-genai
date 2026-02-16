import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'train.csv')

print(f"Loading data from: {file_path}")

# Load the dataset

df = pd.read_csv(file_path, parse_dates=['date'])

# Inspect the data
#print(df.head())
#print(df.dtypes)

item_summary = df.groupby('item')['sales'].mean().sort_values(ascending=False)
print(item_summary.head())
print(item_summary.tail())