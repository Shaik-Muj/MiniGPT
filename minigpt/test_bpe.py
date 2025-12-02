import tiktoken

# Load the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

text = "Hello, world!"

# 1. Encode
tokens = enc.encode(text)
print(f"Tokens: {tokens}")
# Output: [15496, 11, 995, 0] 
# (Notice these numbers are huge compared to our old 0-65 range!)

# 2. Decode
print(f"Decoded: {enc.decode(tokens)}")

# 3. Stats
print(f"Vocab Size: {enc.n_vocab}")
# Output: 50257