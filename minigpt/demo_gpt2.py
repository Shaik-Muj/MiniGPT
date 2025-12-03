import torch
from model import GPTLanguageModel
import tiktoken

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. The Brain Transplant
# This will download ~500MB of weights from OpenAI
model = GPTLanguageModel.from_pretrained('gpt2')
model.to(device)
model.eval()

# 3. The Input
# Let's give it a prompt that requires "Knowledge" (which Shakespeare didn't have)
prompt = "The scientist discovered a new species of"
print(f"\nPrompt: {prompt}")

# 4. Tokenize
enc = tiktoken.get_encoding("gpt2")
input_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

# 5. Generate
print("Generating...")
output_ids = model.generate(input_ids, max_new_tokens=50)
output_text = enc.decode(output_ids[0].tolist())

print("\n--- GPT-2 OUTPUT ---")
print(output_text)