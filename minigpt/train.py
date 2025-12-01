import torch
from model import GPTLanguageModel

# --- HYPERPARAMETERS ---
batch_size = 32       # How many independent sequences to process in parallel
block_size = 8        # Maximum context length for predictions
max_iters = 5000      # Total training steps
eval_interval = 500   # How often to check validation loss
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# --- DATA PIPELINE ---
# 1. Load Data
try:
    # Try the specific path first
    with open(r"minigpt/data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    # Fallback for running locally in the same folder
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

# 2. Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# 3. Train/Val Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 4. Batch Loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i+1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- HELPER: LOSS ESTIMATION ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            # CRITICAL UPDATE: We use _ to ignore the attention maps here
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- MODEL INITIALIZATION ---
model = GPTLanguageModel(vocab_size)
m = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- TRAINING LOOP ---
print("\n--- STARTING TRAINING ---")
for iter in range(max_iters):

    # Monitor progress
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get batch
    xb, yb = get_batch('train')

    # Forward pass
    # CRITICAL UPDATE: We use _ to ignore the attention maps during training
    logits, loss, _ = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- SAVE THE MODEL ---
# We save the weights so we can load them in our visualization script
print("\n--- SAVING MODEL ---")
torch.save(model.state_dict(), 'ckpt.pt')
print("Model saved to ckpt.pt")

# --- FINAL GENERATION ---
print("\n--- GENERATED TEXT ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# Note: The generate function in model.py must also handle the tuple return!
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))