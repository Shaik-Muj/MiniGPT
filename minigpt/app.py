import torch
import tiktoken
import gradio as gr
from model import GPTLanguageModel, GPTConfig

# --- SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 1. Load the Model Architecture
# We must use use_lora=True because the saved weights expect those extra matrices
print("Loading model architecture...")
config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768, use_lora=True)
model = GPTLanguageModel(config)

# 2. Load the Trained Weights
# This includes the OpenAI brain AND your Alpaca LoRA adapters
print("Loading fine-tuned weights...")
checkpoint = torch.load('lora_v2.pt', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 3. Tokenizer
enc = tiktoken.get_encoding("gpt2")

# --- CHAT FUNCTION ---
def generate_response(user_input):
    # A. Format the prompt (The "Social Contract")
    # We must match the format used in finetune.py exactly!
    prompt = f"User: {user_input}\nAssistant:"
    
    # B. Tokenize
    input_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # C. Generate
    # We generate up to 100 new tokens
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, temperature=0.7)
    
    # D. Decode
    generated_text = enc.decode(output_ids[0].tolist())
    
    # E. Clean up the output
    # The model returns the WHOLE prompt + answer. We only want the answer.
    # We split by "Assistant:" and take the second part.
    response = generated_text.split("Assistant:")[-1].strip()
    
    # Optional: Stop if it generates <|endoftext|> or tries to start a new User line
    response = response.split("User:")[0].strip()
    response = response.split("<|endoftext|>")[0].strip()
    
    return response

# --- UI LAUNCHER ---
# This creates the web page
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="MiniGPT Chatbot",
    description="A 124M param GPT-2 model fine-tuned with LoRA on Instruction Data.",
    examples=["What is the capital of France?", "Say hello.", "Explain gravity."]
)

print("Launching App...")
iface.launch()