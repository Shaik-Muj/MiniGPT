from model import GPTConfig, GPTLanguageModel
import torch

# 1. Initialize Standard Model
config = GPTConfig(50257, use_lora=False)
model = GPTLanguageModel(config)
print(f"Standard Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 2. Initialize LoRA Model
config_lora = GPTConfig(50257, use_lora=True)
model_lora = GPTLanguageModel(config_lora)
trainable_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_lora.parameters())

print(f"LoRA Total Params: {total_params}")
print(f"LoRA Trainable Params: {trainable_params}")
print(f"Ratio: {trainable_params/total_params*100:.2f}%")