"""
04_makemore_activations - 学習用サンプル
=========================================
テーマ: NN の「内部状態」を診断して学習を安定させる

4ステップで理解する:
  1. ホッケースティック問題  → W2 の初期化を小さくする
  2. tanh 飽和の可視化      → Kaiming 初期化で解決
  3. 勾配の可視化           → 消失・爆発を数字で確認
  4. Batch Normalization    → 1〜3 を根本的に解決する仕組み

実行方法:
  python sample_activations_batchnorm.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ──────────────────────────────────────
# STEP 0: 共通セットアップ（前回と同じ）
# ──────────────────────────────────────

words = open('../data/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)  # 27

block_size = 3  # コンテキスト長（直前3文字を見る）

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
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr   = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

n_embd  = 10   # 埋め込み次元
n_hidden = 200  # 隠れ層ニューロン数

print(f"語彙サイズ: {vocab_size}, 訓練データ: {Xtr.shape}")

# ──────────────────────────────────────────────────────
# STEP 1: ホッケースティック問題
# ──────────────────────────────────────────────────────
#
# 「何も学習していない初期状態」での理想的なlossは？
#
#   27文字を均等確率で予測すると: -log(1/27) ≈ 3.296
#   これがスタート地点であるべき。
#
# 問題: W2 がランダムのままだと logits がバラバラ
#   → ある1文字に極端な高確率 → それが不正解だと巨大な loss
#   → 最初の数千ステップを「loss を正常値に戻す」だけで無駄消費
#     (グラフが急降下してから平坦になる→ホッケースティック形)
# ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 1: 初期 loss の確認（ホッケースティック問題）")
print("=" * 55)

ideal_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item()
print(f"理想的な初期 loss (均等予測): {ideal_loss:.4f}")

# 検証用ミニバッチ（以降も使い回す）
g = torch.Generator().manual_seed(2147483647)
C_base = torch.randn((vocab_size, n_embd), generator=g)
W1_base = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1_base = torch.zeros(n_hidden)

ix = torch.randint(0, Xtr.shape[0], (32,))
Xb, Yb = Xtr[ix], Ytr[ix]
emb = C_base[Xb]
embcat = emb.view(emb.shape[0], -1)
h_base = torch.tanh(embcat @ W1_base + b1_base)

# ❌ 悪い初期化: W2 がフルスケールのランダム
W2_bad = torch.randn((n_hidden, vocab_size))
b2_bad = torch.randn(vocab_size)
loss_bad = F.cross_entropy(h_base @ W2_bad + b2_bad, Yb)
print(f"❌ 悪い初期化 loss: {loss_bad.item():.4f}  ← 理想値より {loss_bad.item() - ideal_loss:.2f} 高い")

# ✅ 良い初期化: W2 を 0.01 倍に縮小
#   なぜ効く? logits ≈ 0 → softmax が均等確率 → loss ≈ ideal
W2_good = torch.randn((n_hidden, vocab_size)) * 0.01
b2_good = torch.zeros(vocab_size)
loss_good = F.cross_entropy(h_base @ W2_good + b2_good, Yb)
print(f"✅ 良い初期化 loss: {loss_good.item():.4f}  ← 理想値に近い!")


# ──────────────────────────────────────────────────────
# STEP 2: tanh 飽和を「目で見る」
# ──────────────────────────────────────────────────────
#
# tanh の性質:
#   - 入力 |x| > 2 以上 → 出力が ±1 に張り付く（飽和）
#   - 飽和すると tanh の傾き ≈ 0 → 勾配 ≈ 0 → 学習しない
#   - これを「Dead Neuron（死んだニューロン）」と呼ぶ
#
# hpreact（tanh に入る前の値）のヒストグラムで診断できる
#   - 良い状態: hpreact が概ね -2〜+2 の範囲に収まっている
#   - 悪い状態: -5〜+5 など広範囲 → 大半のニューロンが飽和
# ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2: tanh 飽和の可視化")
print("=" * 55)

ix_large = torch.randint(0, Xtr.shape[0], (1000,))
emb_large = C_base[Xtr[ix_large]]
embcat_large = emb_large.view(emb_large.shape[0], -1)

# ❌ W1 スケール大 (std=1.0): hpreact が広がりすぎて飽和
W1_big   = torch.randn((n_embd * block_size, n_hidden)) * 1.0
hpre_big = embcat_large @ W1_big
h_big    = torch.tanh(hpre_big)

# ✅ W1 スケール小 (std=0.1): hpreact が適切な範囲
W1_small   = torch.randn((n_embd * block_size, n_hidden)) * 0.1
hpre_small = embcat_large @ W1_small
h_small    = torch.tanh(hpre_small)

# 飽和率（tanh 出力が ±0.97 以上 = ほぼ飽和）を計算
sat_big   = (h_big.abs() > 0.97).float().mean().item()
sat_small = (h_small.abs() > 0.97).float().mean().item()
print(f"❌ W1 std=1.0 の飽和率: {sat_big*100:.1f}%")
print(f"✅ W1 std=0.1 の飽和率: {sat_small*100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("STEP 2: tanh 飽和の可視化", fontsize=14)

axes[0, 0].hist(hpre_big.view(-1).detach().numpy(), bins=50, color='crimson', alpha=0.8)
axes[0, 0].axvline(-2, color='k', linestyle='--', label='飽和ゾーン境界(±2)')
axes[0, 0].axvline(2, color='k', linestyle='--')
axes[0, 0].set_title(f"❌ hpreact (W1 std=1.0)\n広がりすぎ → 大半が飽和ゾーン")
axes[0, 0].legend()

axes[0, 1].hist(h_big.view(-1).detach().numpy(), bins=50, color='crimson', alpha=0.8)
axes[0, 1].set_title(f"❌ tanh 出力 (飽和率 {sat_big*100:.0f}%)\n±1 に張り付いている")

axes[1, 0].hist(hpre_small.view(-1).detach().numpy(), bins=50, color='steelblue', alpha=0.8)
axes[1, 0].axvline(-2, color='k', linestyle='--')
axes[1, 0].axvline(2, color='k', linestyle='--')
axes[1, 0].set_title(f"✅ hpreact (W1 std=0.1)\nほぼ -2〜+2 に収まっている")

axes[1, 1].hist(h_small.view(-1).detach().numpy(), bins=50, color='steelblue', alpha=0.8)
axes[1, 1].set_title(f"✅ tanh 出力 (飽和率 {sat_small*100:.0f}%)\n中間値に広がっている")

for ax in axes.flat:
    ax.set_xlabel("値")
    ax.set_ylabel("頻度")

plt.tight_layout()
plt.savefig('tanh_saturation.png', dpi=100)
plt.show()
print("→ tanh_saturation.png を保存しました")


# ──────────────────────────────────────────────────────
# STEP 3: Kaiming 初期化（He の初期化）
# ──────────────────────────────────────────────────────
#
# 目標: W1 のスケールを「数学的に正しい値」にする
#
# 直感:
#   n 個の入力を持つ層では、W * x の分散 = n × Var(W)
#   →「分散が層を通しても変わらない」には Var(W) = 1/n
#   → std(W) = 1/sqrt(n)
#
# tanh 用の補正:
#   tanh は入力を圧縮するため、gain = 5/3 を掛けて補正
#   → std = (5/3) / sqrt(fan_in)
#
# fan_in = 入力ニューロン数 = n_embd * block_size = 10 * 3 = 30
# ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 3: Kaiming 初期化")
print("=" * 55)

fan_in = n_embd * block_size  # 30
gain   = 5 / 3                 # tanh 用ゲイン（経験則）
kaiming_std = gain / (fan_in ** 0.5)

print(f"fan_in:      {fan_in}")
print(f"gain:        {gain:.4f}  (tanh の場合は 5/3 が経験則)")
print(f"Kaiming std: {kaiming_std:.4f}")

W1_kaiming   = torch.randn((n_embd * block_size, n_hidden)) * kaiming_std
hpre_kaiming = embcat_large @ W1_kaiming
h_kaiming    = torch.tanh(hpre_kaiming)

sat_kaiming = (h_kaiming.abs() > 0.97).float().mean().item()
print(f"\nKaiming 初期化の飽和率: {sat_kaiming*100:.1f}%")
print(f"hpreact の標準偏差:    {hpre_kaiming.std().item():.4f}  (理想: ≈ 1.0)")
print(f"h の標準偏差:          {h_kaiming.std().item():.4f}")


# ──────────────────────────────────────────────────────
# STEP 4: 勾配を「見る」
# ──────────────────────────────────────────────────────
#
# backward() の後に p.grad を確認する。
#
# 健全な状態:
#   - grad の絶対値の平均が 1e-3 〜 1e-1 くらい
#   - 全層でだいたい同じオーダー
#
# 問題のあるパターン:
#   - grad ≈ 0          → 勾配消失（学習しない）
#   - grad >> 1         → 勾配爆発（学習が発散）
#   - 層によってオーダーが大きく違う → 一部だけ学習が進まない
# ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4: 勾配の可視化")
print("=" * 55)

# 改善後のパラメータで 1 ステップ計算
g_seed = torch.Generator().manual_seed(2147483647)
C2  = torch.randn((vocab_size, n_embd), generator=g_seed, requires_grad=True)
W12 = torch.randn((n_embd * block_size, n_hidden), generator=g_seed) * kaiming_std
W12.requires_grad_(True)
b12 = torch.zeros(n_hidden, requires_grad=True)
W22 = torch.randn((n_hidden, vocab_size), generator=g_seed) * 0.01
W22.requires_grad_(True)
b22 = torch.zeros(vocab_size, requires_grad=True)

params_named = [('C', C2), ('W1', W12), ('b1', b12), ('W2', W22), ('b2', b22)]

emb2    = C2[Xb]
embcat2 = emb2.view(emb2.shape[0], -1)
h2      = torch.tanh(embcat2 @ W12 + b12)
loss2   = F.cross_entropy(h2 @ W22 + b22, Yb)
loss2.backward()

print(f"{'パラメータ':<6} | {'grad 平均(絶対値)':<18} | {'grad 標準偏差':<14} | 診断")
print("-" * 65)
for name, p in params_named:
    g_mean = p.grad.abs().mean().item()
    g_std  = p.grad.std().item()
    if g_mean < 1e-5:
        diag = "⚠ 勾配消失"
    elif g_mean > 1.0:
        diag = "⚠ 勾配爆発"
    else:
        diag = "✅ 正常"
    print(f"{name:<6} | {g_mean:<18.6f} | {g_std:<14.6f} | {diag}")


# ──────────────────────────────────────────────────────
# STEP 5: Batch Normalization（バッチ正規化）
# ──────────────────────────────────────────────────────
#
# 問題: Kaiming 初期化は「学習の最初」だけ安定させる。
#       学習が進むにつれ、内部の分布はまたズレていく。
#
# BatchNorm の発想:
#   「hpreact を毎回強制的に mean=0, std=1 に正規化してしまえばいい」
#
# 実装:
#   bnmean = hpreact.mean(dim=0)  ← バッチ方向（行方向）の平均
#   bnstd  = hpreact.std(dim=0)   ← バッチ方向の標準偏差
#   hpreact_norm = (hpreact - bnmean) / (bnstd + ε)
#
# gamma（スケール）と beta（シフト）の役割:
#   正規化しすぎると tanh の中心付近しか使えなくなり表現力が落ちる。
#   gamma と beta は「必要なら正規化を緩める」学習可能パラメータ。
#   初期値: gamma=1, beta=0 → まず完全正規化からスタート
#
# 推論時の注意:
#   バッチが 1 サンプルだと mean/std が計算できない。
#   → 学習中に「移動平均」を記録しておき、推論時はそれを使う。
# ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5: Batch Normalization の仕組み")
print("=" * 55)

# 小さい例で BatchNorm の挙動を確認
hpre_demo = torch.randn(32, 10) * 5 + 3  # 意図的に mean=3, std=5 にする
print(f"正規化前: mean={hpre_demo.mean():.3f}, std={hpre_demo.std():.3f}")

bn_mean = hpre_demo.mean(0, keepdim=True)   # shape: (1, 10)
bn_std  = hpre_demo.std(0, keepdim=True)    # shape: (1, 10)
eps = 1e-5
hpre_normed = (hpre_demo - bn_mean) / (bn_std + eps)
print(f"正規化後: mean={hpre_normed.mean():.3f}, std={hpre_normed.std():.3f}")

bngain = torch.ones((1, 10))   # gamma (初期: スケール変えない)
bnbias = torch.zeros((1, 10))  # beta  (初期: シフトしない)
hpre_bn = bngain * hpre_normed + bnbias
print(f"gamma/beta 適用後: mean={hpre_bn.mean():.3f}, std={hpre_bn.std():.3f}")
print("→ 初期状態では正規化そのまま。学習が進むと gamma/beta が調整される。")


# ──────────────────────────────────────────────────────
# STEP 6: 全改善を組み合わせた最終トレーニング
# ──────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 6: 全改善を組み合わせた学習")
print("=" * 55)

g3 = torch.Generator().manual_seed(2147483647)
C3  = torch.randn((vocab_size, n_embd), generator=g3)
W13 = torch.randn((n_embd * block_size, n_hidden), generator=g3) * kaiming_std  # Kaiming
b13 = torch.zeros(n_hidden)
W23 = torch.randn((n_hidden, vocab_size), generator=g3) * 0.01  # 小さい初期値
b23 = torch.zeros(vocab_size)

# BatchNorm の学習パラメータ
bngain3 = torch.ones((1, n_hidden))
bnbias3 = torch.zeros((1, n_hidden))

# 推論時に使う移動平均（学習中に更新していく）
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running  = torch.ones((1, n_hidden))

parameters3 = [C3, W13, b13, W23, b23, bngain3, bnbias3]
for p in parameters3:
    p.requires_grad_(True)

print(f"総パラメータ数: {sum(p.nelement() for p in parameters3):,}")

max_steps = 10000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g3)
    Xb3, Yb3 = Xtr[ix], Ytr[ix]

    # 順伝播
    emb3    = C3[Xb3]
    embcat3 = emb3.view(emb3.shape[0], -1)
    hpreact3 = embcat3 @ W13 + b13

    # BatchNorm（学習時）
    bnmean_i = hpreact3.mean(0, keepdim=True)
    bnstd_i  = hpreact3.std(0, keepdim=True)
    hpreact3 = bngain3 * (hpreact3 - bnmean_i) / (bnstd_i + 1e-5) + bnbias3

    # 移動平均を更新（推論時に使う）
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean_i
        bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstd_i

    h3     = torch.tanh(hpreact3)
    logits3 = h3 @ W23 + b23
    loss3   = F.cross_entropy(logits3, Yb3)

    # 逆伝播
    for p in parameters3:
        p.grad = None
    loss3.backward()

    # 更新（後半は学習率を下げる）
    lr = 0.1 if i < 5000 else 0.01
    for p in parameters3:
        p.data -= lr * p.grad

    lossi.append(loss3.log10().item())
    if i % 2000 == 0:
        print(f"step {i:6d}/{max_steps}: loss {loss3.item():.4f}")

# 損失グラフ
plt.figure(figsize=(10, 4))
plt.plot(torch.tensor(lossi).view(-1, 100).mean(1).numpy())
plt.xlabel("ステップ (×100)")
plt.ylabel("log10(loss)")
plt.title("学習曲線（BatchNorm + Kaiming Init + W2小初期化）")
plt.grid(True)
plt.savefig('training_loss_batchnorm.png', dpi=100)
plt.show()
print("→ training_loss_batchnorm.png を保存しました")


# 評価（推論時は移動平均を使う）
@torch.no_grad()
def evaluate(X, Y):
    emb = C3[X]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W13 + b13
    # 推論時: 移動平均を使って正規化
    hpreact = bngain3 * (hpreact - bnmean_running) / (bnstd_running + 1e-5) + bnbias3
    h = torch.tanh(hpreact)
    return F.cross_entropy(h @ W23 + b23, Y).item()

print(f"\n訓練ロス: {evaluate(Xtr, Ytr):.4f}")
print(f"検証ロス: {evaluate(Xdev, Ydev):.4f}")

# ──────────────────────────────────────────────────────
# まとめ
# ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("まとめ")
print("=" * 55)
print("""
問題                   原因                    解決策
─────────────────────────────────────────────────────
ホッケースティック    W2 が大きすぎる        W2 *= 0.01
tanh 飽和             W1 が大きすぎる        Kaiming 初期化
勾配消失              上2つの組み合わせ      上記2つで改善
分布のズレ（学習中）  初期化だけでは限界     Batch Normalization

BatchNorm が解決するもの:
  - 初期化が悪くても内部分布を毎回リセット
  - 深いネットワークでも安定して学習できる
  - 学習率を大きめにできる（= 学習が速くなる）
""")
