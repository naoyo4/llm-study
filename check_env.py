import sys
import torch
import numpy as np
import tiktoken

print("=" * 50)
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # False で正常
print(f"CPU threads: {torch.get_num_threads()}")
print("=" * 50)

# 簡単なテンソル演算
x = torch.randn(3, 4)
y = torch.randn(4, 5)
z = x @ y
print(f"Matmul test: {x.shape} @ {y.shape} = {z.shape}")

# 自動微分テスト
a = torch.tensor(2.0, requires_grad=True)
b = a ** 3 + 2 * a
b.backward()
print(f"Autograd test: d(a^3 + 2a)/da at a=2 → {a.grad.item()} (期待値: 14.0)")

# トークナイザ
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, LLM world!")
print(f"Tokenizer test: 'Hello, LLM world!' → {tokens}")
print(f"Decoded: {enc.decode(tokens)}")

print("=" * 50)
print("✅ 環境OK")
