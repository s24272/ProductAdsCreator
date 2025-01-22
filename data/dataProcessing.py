from datasets import load_dataset
import pandas as pd

ds = load_dataset("FourthBrainGenAI/Product-Descriptions-and-Ads")
df = pd.DataFrame(ds['train'])
df.to_csv('data/amazon_products.csv', index=False)