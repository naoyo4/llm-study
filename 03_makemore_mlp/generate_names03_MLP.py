"""
MLP 言語モデルで名前を生成し、確率スコア上位50を表示する
実行: python generate_names.py
"""

import torch
import torch.nn.functional as F
import random

# ── データ準備 ──────────────────────────────────────
words = open('../data/names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8*len(words)), int(0.9*len(words))
Xtr, Ytr   = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

# ── モデル初期化（BatchNorm + Kaiming 版）──────────
n_embd, n_hidden = 10, 200
fan_in = n_embd * block_size
kaiming_std = (5/3) / (fan_in**0.5)

g = torch.Generator().manual_seed(2147483647)
C      = torch.randn((vocab_size, n_embd),  generator=g)
W1     = torch.randn((fan_in, n_hidden),     generator=g) * kaiming_std
b1     = torch.zeros(n_hidden)
W2     = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2     = torch.zeros(vocab_size)
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_run = torch.zeros((1, n_hidden))
bnstd_run  = torch.ones((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad_(True)

# ── 学習 ────────────────────────────────────────────
print("学習中...", flush=True)
max_steps, batch_size = 30000, 32

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb     = C[Xb]
    embcat  = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1

    bnmean_i = hpreact.mean(0, keepdim=True)
    bnstd_i  = hpreact.std(0, keepdim=True)
    hpreact  = bngain * (hpreact - bnmean_i) / (bnstd_i + 1e-5) + bnbias

    with torch.no_grad():
        bnmean_run = 0.999*bnmean_run + 0.001*bnmean_i
        bnstd_run  = 0.999*bnstd_run  + 0.001*bnstd_i

    h      = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss   = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 15000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    if i % 10000 == 9999:
        print(f"  step {i+1}/{max_steps}: loss {loss.item():.4f}")

# ── 評価 ────────────────────────────────────────────
@torch.no_grad()
def eval_loss(X, Y):
    emb = C[X]
    ec  = emb.view(emb.shape[0], -1)
    hp  = ec @ W1 + b1
    hp  = bngain * (hp - bnmean_run) / (bnstd_run + 1e-5) + bnbias
    h   = torch.tanh(hp)
    return F.cross_entropy(h @ W2 + b2, Y).item()

print(f"訓練 loss: {eval_loss(Xtr,Ytr):.4f}  検証 loss: {eval_loss(Xdev,Ydev):.4f}")

# ── 名前生成（log 確率でスコアリング）──────────────
@torch.no_grad()
def generate_with_score(seed):
    g2      = torch.Generator().manual_seed(seed)
    context = [0] * block_size
    name    = ''
    log_prob = 0.0
    for _ in range(20):
        emb    = C[torch.tensor([context])]
        ec     = emb.view(1, -1)
        hp     = ec @ W1 + b1
        hp     = bngain * (hp - bnmean_run) / (bnstd_run + 1e-5) + bnbias
        h      = torch.tanh(hp)
        logits = h @ W2 + b2
        probs  = F.softmax(logits, dim=1)
        ix     = torch.multinomial(probs, 1, generator=g2).item()
        log_prob += torch.log(probs[0, ix]).item()
        if ix == 0:
            break
        name    += itos[ix]
        context  = context[1:] + [ix]
    avg_log_prob = log_prob / max(len(name), 1)
    return name, avg_log_prob

print("\n名前を生成中...")
candidates = {}
for seed in range(5000):
    name, score = generate_with_score(seed)
    if 2 <= len(name) <= 10 and name not in candidates:
        candidates[name] = score

top50 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:50]

print("\n── 確率スコア上位50の生成名 ──")
print(f"{'順位':<4} {'名前':<14} {'スコア(文字あたり平均log確率)'}")
print("-" * 46)
for rank, (name, score) in enumerate(top50, 1):
    bar = '█' * int(-score * 2)
    print(f"{rank:<4} {name.capitalize():<14} {score:.4f}  {bar}")
