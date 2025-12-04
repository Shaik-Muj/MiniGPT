import torch
import json
import os
import tiktoken
import urllib.request
from model import GPTLanguageModel, GPTConfig

# --- CONFIGURATION ---
N_SAMPLES = 1000      # 30 is too low. 1000 is the "Sweet Spot" for a 15 min demo run.
MAX_ITERS = 800       # How many training steps (updates)
BATCH_SIZE = 4        # Lower this to 2 or 1 if you run out of memory
LEARNING_RATE = 3e-4  
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"--- SETUP ---")
print(f"Device: {device}")
print(f"Target Samples: {N_SAMPLES}")

# --- 1. DOWNLOAD REAL DATA (Alpaca Cleaned) ---
data_path = "alpaca_data_cleaned.json"
url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"

if not os.path.exists(data_path):
    print(f"Downloading {data_path}...")
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r") as f:
    full_data = json.load(f)

# Slice the dataset to your desired size
# We take the first N_SAMPLES
dataset = full_data[:N_SAMPLES]
print(f"Loaded {len(dataset)} examples from Alpaca.")

# --- 2. FORMAT & TOKENIZE ---
def format_prompt(entry):
    # This matches the format we used in app.py
    return f"User: {entry['instruction']}\nAssistant: {entry['output']}<|endoftext|>"

enc = tiktoken.get_encoding("gpt2")
data_tokens = []

print("Tokenizing data (this might take a moment)...")
for entry in dataset:
    text = format_prompt(entry)
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    data_tokens.extend(tokens)

data_tensor = torch.tensor(data_tokens, dtype=torch.long)
print(f"Total Tokens in Dataset: {len(data_tensor)}")

def get_batch():
    # Randomly grab chunks of 128 tokens
    ix = torch.randint(len(data_tensor) - 128, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i : i + 128] for i in ix])
    y = torch.stack([data_tensor[i+1 : i + 128 + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- 3. LOAD MODEL (Brain Transplant) ---
print("Loading OpenAI GPT-2 with LoRA adapters...")
# Force use_lora=True
model = GPTLanguageModel.from_pretrained('gpt2')
model.to(device)

# --- 4. FREEZE BASE MODEL ---
print("Freezing base weights...")
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# --- 5. TRAINING LOOP ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"\n--- STARTING TRAINING ({MAX_ITERS} steps) ---")
model.train()
for iter in range(MAX_ITERS):
    xb, yb = get_batch()
    
    # Forward pass (passing targets=yb is CRITICAL)
    logits, loss, _ = model(xb, targets=yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 10 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# --- 6. SAVE ---
print("\n--- SAVING REAL MODEL ---")
# Saving as a new name so we don't overwrite your test file
torch.save(model.state_dict(), 'lora_v2.pt')
print("Saved to lora_v2.pt")