"""
Sample from a trained model
"""
import os
import pickle
import torch
import tiktoken
from contextlib import nullcontext
from model import GPTConfig, GPT

# -------------------- CONFIG --------------------
out_dir = 'out-eunji'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
start = "은지: "   # 시작 프롬프트
num_samples = 3
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 42
device = 'cpu'
compile = False

# -------------------- SETUP ---------------------
torch.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext()

# -------------------- LOAD CHECKPOINT --------------------
checkpoint = torch.load(ckpt_path, map_location='cpu')

# -------------------- LOAD MODEL --------------------
gptconf = GPTConfig(
    n_layer=4,
    n_head=4,
    n_embd=128,
    block_size=64,
    bias=True,
    vocab_size=50257   # ✅ 여기를 꼭 바꿔야 해!
)

if compile:
    model = torch.compile(model)

# -------------------- ENCODER --------------------
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# -------------------- PROMPT --------------------
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# -------------------- GENERATE --------------------
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')




