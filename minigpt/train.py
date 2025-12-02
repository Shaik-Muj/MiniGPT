import torch
import tiktoken
from model import GPTLanguageModel, GPTConfig

# --- HYPERPARAMETERS ---
batch_size = 32       
block_size = 8        
max_iters = 5000      
eval_interval = 500   
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# --- DATA PIPELINE ---
# 1. Load Data
try:
    with open(r"minigpt/data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

# We no longer calculate unique characters. We use the pre-trained GPT-2 tokenizer.
enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab 

# allowed_special ensures it handles special tokens like <|endoftext|> if they appear
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

print(f"Vocab Size: {vocab_size}") 
# ------------------------------------------

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
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- MODEL INITIALIZATION ---
# We keep n_layer=3, n_head=4, n_embd=64 for now to keep training fast on your laptop
# But the ARCHITECTURE is now identical to the big GPT-2
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=4, 
    n_head=4, 
    n_embd=128 # Increased slightly for BPE
)
model = GPTLanguageModel(config)
m = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- TRAINING LOOP ---
print("\n--- STARTING TRAINING ---")
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss, _ = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- SAVE THE MODEL ---
print("\n--- SAVING MODEL ---")
torch.save(model.state_dict(), 'ckpt.pt')
print("Model saved to ckpt.pt")

# --- FINAL GENERATION ---
print("\n--- GENERATED TEXT ---")
# 198 is the standard GPT-2 token ID for \n
start_token = torch.tensor([[198]], dtype=torch.long, device=device)
print(decode(m.generate(start_token, max_new_tokens=500)[0].tolist()))