import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/amazon_products_with_detailed_categories.csv')

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

test_df, valid_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv('data/products_category_train.csv', index=False)
test_df.to_csv('data/products_category_test.csv', index=False)
valid_df.to_csv('data/products_category_valid.csv', index=False)

print("Dataset został podzielony i zapisany")
