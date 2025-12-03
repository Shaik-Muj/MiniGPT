import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer # The frozen pre-trained layer
        self.rank = rank
        self.alpha = alpha
        
        # Freezing the pre-trained layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # LoRA Matrices: A (In) and B (Out)
        # n_in -> rank -> n_out
        n_in = self.linear.in_features
        n_out = self.linear.out_features
        
        self.lora_a = nn.Parameter(torch.zeros(n_in, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, n_out))
        
        # Scaling factor
        self.scaling = self.alpha / self.rank
        
        # Initialization (Critical for LoRA)
        # A is Gaussian, B is Zero. 
        # Result: A*B starts at 0, so we start exactly as the pre-trained model.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        # 1. The Regular Path (Frozen)
        regular_out = self.linear(x)
        
        # 2. The LoRA Path (Trainable)
        # x @ A @ B
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scaling
        
        # 3. Sum them up
        return regular_out + lora_out

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 1. Create the standard linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # 2. [NEW] Wrap it in LoRA if config asks for it
        # We check if 'use_lora' exists in config, default to False
        if getattr(config, 'use_lora', False):
            self.c_attn = LoRALinear(self.c_attn, rank=8, alpha=16)
            
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        
        # Everything else remains EXACTLY the same...
        # The self.c_attn(x) call now automatically uses the LoRA forward logic
        
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2) 
            v = torch.cat((past_value, v), dim=-2)
            
        present = (k, v)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        total_len = k.size(-2)
        att = att.masked_fill(self.bias[:,:,total_len-T:total_len, :total_len] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y, present

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None):
        # We must pipe 'layer_past' into attention and get 'present' back
        attn_out, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = kwargs.get('block_size', 1024)
        self.n_embd = kwargs.get('n_embd', 768)
        self.n_head = kwargs.get('n_head', 12)
        self.n_layer = kwargs.get('n_layer', 12)
        self.dropout = kwargs.get('dropout', 0.1)
        self.use_lora = kwargs.get('use_lora', False)

class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

        # [NEW] Global Freezing Logic for LoRA
        # If we are using LoRA, we must freeze EVERYTHING that isn't a LoRA parameter.
        if getattr(config, 'use_lora', False):
            print("LoRA detected: Freezing base model weights...")
            for name, param in self.named_parameters():
                if 'lora' in name:
                    # These are our Adapter matrices (A and B). Train them!
                    param.requires_grad = True 
                else:
                    # This is the pre-trained brain. Freeze it!
                    param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kv=None, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # If past_kv is provided, we only need to embed the LAST token
        # But we need to keep track of the position index correctly
        if past_kv is not None:
            # The position is: total_length_so_far - 1
            # past_kv[0][0] is (k,v) for layer 0. k shape is (B, nh, T_past, hs)
            past_length = past_kv[0][0].size(-2)
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = tok_emb + pos_emb
        
        new_kv = []
        for i, block in enumerate(self.transformer.h):
            # Pass the specific layer's past history
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_present = block(x, layer_past=layer_past)
            new_kv.append(layer_present)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return logits, loss, new_kv

    # [CHANGE] Updated Generate function to use the cache
    def generate(self, idx, max_new_tokens):
        past_kv = None
        for _ in range(max_new_tokens):
            # If we have cache, we only feed the LAST token
            if past_kv is not None:
                idx_cond = idx[:, -1:] 
            else:
                idx_cond = idx
            
            logits, _, past_kv = self(idx_cond, past_kv=past_kv)
            
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print(f"Loading weights for {model_type}...")
        config = GPTConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
        model = GPTLanguageModel(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith('attn.masked_bias')] 
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] 

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_hf_keys) == len(sd_keys), f"mismatched keys: {len(sd_hf_keys)} != {len(sd_keys)}"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model