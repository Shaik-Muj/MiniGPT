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


# Constructing a small test model to ensure everything is working
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) # has a shape of (B,T,C)

        if targets==None:
            loss = None
        
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print("\n--- MODEL INITIALIZATION ---")
model = BigramLanguageModel(vocab_size)
m = model.to('cpu')

out, loss = m(xb, yb)
print(f"Model logits shape: {out.shape}")
print(f"Initial Loss (should be high): {loss.item()}")

print("\n--- GENERATING TEXT (UNTRAINED) ---")
# Start with a single zero (newline character usually) as context
idx = torch.zeros((1, 1), dtype=torch.long) 

# Ask the model to generate 100 characters
generated_tokens = m.generate(idx, max_new_tokens=100)
generated_text = decode(generated_tokens[0].tolist())

print(generated_text)