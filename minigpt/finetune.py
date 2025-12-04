import torch
import json
import os
import tiktoken
from model import GPTLanguageModel, GPTConfig

# --- SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
learning_rate = 3e-4 # LoRA usually likes a slightly higher LR than full finetuning
max_iters = 100 # Quick demo run (increase to 500+ for real results)
batch_size = 4

# --- 1. PREPARE DATA (Alpaca-Style) ---
data_path = "alpaca_dummy.json"

# Create a small dummy dataset of Instructions
dummy_data = [
    {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
    {"instruction": "Who wrote Romeo and Juliet?", "output": "William Shakespeare wrote Romeo and Juliet."},
    {"instruction": "Convert 100c to fahrenheit.", "output": "100 degrees Celsius is equal to 212 degrees Fahrenheit."},
    {"instruction": "Explain gravity.", "output": "Gravity is a fundamental force that attracts objects with mass towards each other."},
    {"instruction": "Say hello.", "output": "Hello! How can I help you today?"},
] * 20 # Duplicate to make a "dataset"

with open(data_path, "w") as f:
    json.dump(dummy_data, f)

# Formatter: Enforces the "User/Assistant" Social Contract
def format_prompt(entry):
    return f"User: {entry['instruction']}\nAssistant: {entry['output']}<|endoftext|>"

# Tokenize
enc = tiktoken.get_encoding("gpt2")
data_tokens = []
print("Tokenizing data...")
for entry in dummy_data:
    text = format_prompt(entry)
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    data_tokens.extend(tokens)

data_tensor = torch.tensor(data_tokens, dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data_tensor) - 128, (batch_size,))
    x = torch.stack([data_tensor[i : i + 128] for i in ix])
    y = torch.stack([data_tensor[i+1 : i + 128 + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- 2. LOAD PRE-TRAINED MODEL + LoRA ---
print("Loading OpenAI GPT-2 with LoRA adapters...")
# This calls your PATCHED logic from model.py
model = GPTLanguageModel.from_pretrained('gpt2')
model.to(device)

# --- 3. FREEZE EVERYTHING EXCEPT LoRA ---
# (Double check to be safe, though model.py handles this)
print("Freezing base model...")
trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    if 'lora' in name:
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

print(f"Trainable Parameters: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")

# --- 4. TRAINING LOOP ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n--- STARTING INSTRUCTION TUNING ---")
model.train()
for iter in range(max_iters):
    xb, yb = get_batch()
    
    # Forward pass
    logits, loss, _ = model(xb, targets=yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 10 == 0:
        print(f"Step {iter}: Loss {loss.item():.4f}")

# --- 5. SAVE ADAPTER ---
print("\n--- SAVING LoRA ADAPTER ---")
# In a real library, we'd only save the LoRA weights. 
# Here, for simplicity, we save the whole state dict (which includes the frozen weights).
# Ideally, you filter for keys containing 'lora' before saving to save space.
torch.save(model.state_dict(), 'lora_finetuned.pt')
print("Saved to lora_finetuned.pt")