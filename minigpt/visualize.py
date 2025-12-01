import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model import GPTLanguageModel

# --- SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Re-create Tokenizer, this matches the training one exactly
# We need to reload the text just to get the same vocab mapping
try:
    with open(r"minigpt/data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# 2. Load the Model
print("Loading model...")
model = GPTLanguageModel(vocab_size)
# Load the weights we saved in train.py
model.load_state_dict(torch.load('ckpt.pt', map_location=device))
model.to(device)
model.eval() # Switch to evaluation mode

# --- VISUALIZATION FUNCTION ---
def visualize_attention(input_str):
    # Prepare input
    x = torch.tensor(encode(input_str), dtype=torch.long, device=device).unsqueeze(0) # Add batch dim
    
    # Run forward pass (we only care about the 3rd return value: attentions)
    with torch.no_grad():
        _, _, attentions = model(x)
    
    # attentions shape: (Batch, Num_Heads, Time, Time)
    # We will visualize the FIRST head of the LAST layer
    # shape becomes (Time, Time)
    att_matrix = attentions[0, 0, :, :].cpu().numpy()
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_matrix, xticklabels=list(input_str), yticklabels=list(input_str), cmap="viridis")
    plt.title("Attention Map (Layer 3, Head 1)")
    plt.xlabel("Key (Looking at...)")
    plt.ylabel("Query (Current Token)")
    plt.show()

# --- RUN IT ---
# Let's test it with a classic line.
# Note: The input must be shorter than block_size (8) for a clear diagonal, 
# or we just crop to the last 8 chars.
test_sentence = "Thou art"
print(f"Visualizing attention for: '{test_sentence}'")
visualize_attention(test_sentence)