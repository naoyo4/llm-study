"""
Bigram 言語モデルで名前を生成し、確率スコア上位50を表示する
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

# Bigram: 直前1文字 → 次の1文字を予測
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"語彙サイズ: {vocab_size}")
print(f"訓練サンプル数: {xs.shape[0]}")

# ── モデル初期化 ─────────────────────────────────────
# 重み行列 W 1枚だけ（27×27）
# W[i, j] ≈ 文字 i の次に文字 j が来る「スコア」
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

# ── 学習 ────────────────────────────────────────────
print("\n学習中...", flush=True)
for i in range(200):
    xenc   = F.one_hot(xs, num_classes=vocab_size).float()
    logits = xenc @ W
    loss   = F.cross_entropy(logits, ys)
    W.grad = None
    loss.backward()
    W.data -= 50 * W.grad
    if i % 50 == 49:
        print(f"  step {i+1}/200: loss {loss.item():.4f}")

print(f"最終 loss: {loss.item():.4f}")
print("（参考: MLP モデルの最終 loss ≈ 2.16）")

# ── 名前生成（log 確率でスコアリング）──────────────
@torch.no_grad()
def generate_with_score(seed):
    g2       = torch.Generator().manual_seed(seed)
    ix       = 0   # '.' からスタート
    name     = ''
    log_prob = 0.0
    for _ in range(20):
        xenc   = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        probs  = F.softmax(logits, dim=1)
        ix     = torch.multinomial(probs, 1, generator=g2).item()
        log_prob += torch.log(probs[0, ix]).item()
        if ix == 0:
            break
        name += itos[ix]
    avg_log_prob = log_prob / max(len(name), 1)
    return name, avg_log_prob

print("\n名前を生成中...")
candidates = {}
for seed in range(5000):
    name, score = generate_with_score(seed)
    if 2 <= len(name) <= 10 and name not in candidates:
        candidates[name] = score

top50 = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:50]

print("\n── Bigram モデル: 確率スコア上位50 ──")
print(f"{'順位':<4} {'名前':<14} {'スコア(文字あたり平均log確率)'}")
print("-" * 46)
for rank, (name, score) in enumerate(top50, 1):
    bar = '█' * int(-score * 2)
    print(f"{rank:<4} {name.capitalize():<14} {score:.4f}  {bar}")
