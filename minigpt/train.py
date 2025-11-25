import torch

# reading the input.txt file
with open("minigpt\data\input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# creating the vocabulary
chars = list(sorted(set(text)))
vocab_size = len(chars)

# create mappings (the tokenizer)
stoi = {ch:i for i,ch in enumerate(chars)} # string to integer
itos = {i:ch for i,ch in enumerate(chars)} # integer to string

encode = lambda s : [stoi[c] for c in s] # string to integer 
decode = lambda l : ''.join([itos[i] for i in l])

print(f"Length of dataset: {len(text)} characters")
print(f"Vocab size: {vocab_size}")
print(f"First 50 chars: {text[:50]}")
print(f"First 50 tokens: {encode(text[:50])}")
print(f"First 50 decoded: {decode(encode(text[:50]))}")

# encode the entire text
data = torch.tensor(encode(text), dtype=torch.long)

# splitting into training and validation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# batching 
block_size = 8
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i+1 : i + block_size + 1] for i in ix])

    return x, y

# visualizing a single batch to understand better
xb, yb = get_batch("train")
print("\n--- BATCH VISUALIZATION ---")
print(f"Inputs (x) shape: {xb.shape}")
print(f"Inputs (x):\n{xb}")
print(f"Targets (y):\n{yb}")

print("\n--- WHAT THE MODEL LEARNS ---")
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {context.tolist()} -> Target is {target}")
    break
