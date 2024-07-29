import pandas as pd

# Read the Parquet file
df = pd.read_parquet('dataset.parquet')

# Convert the DataFrame to JSONL
df.to_json('shisa-v1.jsonl', orient='records', lines=True)
