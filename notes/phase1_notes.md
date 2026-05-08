# LLM学習ノート

## Phase 1: Karpathy "Neural Networks: Zero to Hero"

### 01_micrograd（自動微分）

**キーコンセプト**
- 各 `Value` オブジェクトが「どの演算で生まれたか」を記憶する
- `backward()` はトポロジカルソートした順の逆順で `_backward` を呼ぶ
- `grad +=` である理由: 同じ変数が複数回使われると勾配が「足し合わさる」（連鎖律の多変数版）

**確認問題メモ**
<!-- ここに自分の答えを書いていく -->

---

### 02_makemore_bigram
（進めたら記録する）
