import os
from tokenizer import Tokenizer
import torch
import json
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'tiny-stories/data00.json')
with open(input_file_path) as file:
    raw = json.load(file)
    data = [element.get('story').replace('\n', ' ').lstrip() for element in raw]

# ========== Encode raw data to get unique token ids
tokenizer = Tokenizer()

# get unique tokens
token_ids = [tokenizer.encode(batch) for batch in data if len(batch) > 0]
# print(f"Max token lenght: {max(len(sub_arr) for sub_arr in token_ids)}")

flatten = [item for batch in token_ids for item in batch]
unique_token_ids = list(set(flatten))

print(f"Shortest batch: {np.min([len(batch) for batch in token_ids])}")
print(f"original tokenizer vocab size: {tokenizer.n_vocab}")
print(f"reduced tokenizer vocab size: {len(unique_token_ids)}")

tokenizer.set_unique_tokens(unique_token_ids)

# ============ Train and Test splits
n = len(data)
train_batches = data[:int(n*0.9)]
val_batches = data[int(n*0.9):]

# encode to integers
train_ids = [tokenizer.encode(batch) for batch in train_batches if len(batch) > 0]
val_ids = [tokenizer.encode(batch) for batch in val_batches if len(batch) > 0]
print(f"train has {len(train_ids):,} batches")
print(f"val has {len(val_ids):,} batches")

# store as bin files
torch.save(unique_token_ids, 'unique_token_ids.pt')
torch.save(train_ids, 'train_ids.pt')
torch.save(val_ids, 'val_ids.pt')
