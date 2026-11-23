"""次回の本数字を予測するモデルのウォークフォワード・バックテスト。

各回 t について「第 t 回より前のデータだけ」で学習し、37 個の数字それぞれが
第 t 回の本数字に入る確率を予測する。予測上位 7 個を 1 口として買ったときの
一致数を、完全ランダムに選んだ場合の理論値と比べる。

ランダムに 7 個選んだときの一致数は超幾何分布に従い、平均 7*7/37 ≈ 1.324 個。
モデルがこれを統計的に有意に上回らなければ「予測できていない」と判断する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .data import N_NUMBERS, N_PICK, ROOT, indicator_matrix, load
from .ml import MLP, LogisticRegression

BASE_RATE = N_PICK / N_NUMBERS
WINDOWS = (5, 10, 20, 50, 100)
REPORT_DIR = ROOT / "reports"


# ---------------------------------------------------------------- 特徴量

def number_features(X: np.ndarray) -> np.ndarray:
    """(T, 37, F) の特徴量。t 行目は第 t 回 *より前* の情報だけから作る。

    - 直近 w 回の出現率 (w = 5, 10, 20, 50, 100)
    - 通算出現率
    - 最後に出てからの経過回数 (log)
    - 前回出たかどうか
    - 数字そのもの (1〜37 を 0〜1 に正規化)
    """
    T = X.shape[0]
    cs = np.vstack([np.zeros((1, N_NUMBERS)), np.cumsum(X, axis=0)])  # cs[t] = 第 t 回より前の合計
    feats = []
    for w in WINDOWS:
        lo = np.maximum(np.arange(T) - w, 0)
        n = np.maximum(np.arange(T) - lo, 1)[:, None]
        feats.append((cs[np.arange(T)] - cs[lo]) / n)
    n_all = np.maximum(np.arange(T), 1)[:, None]
    feats.append(cs[np.arange(T)] / n_all)

    gap = np.zeros((T, N_NUMBERS))
    last = np.full(N_NUMBERS, -1)
    for t in range(T):
        gap[t] = np.where(last >= 0, t - last, t + 1)
        last[X[t] == 1] = t
    feats.append(np.log1p(gap))

    prev = np.vstack([np.zeros((1, N_NUMBERS)), X[:-1]])
    feats.append(prev)
    feats.append(np.tile(np.arange(N_NUMBERS) / (N_NUMBERS - 1), (T, 1)))
    return np.stack(feats, axis=-1)


# ---------------------------------------------------------------- モデル

class Model:
    name = "base"
    retrain_every = 1

    def fit(self, F: np.ndarray, X: np.ndarray, t_end: int) -> None:  # 第 t_end 回より前で学習
        pass

    def predict(self, F: np.ndarray, X: np.ndarray, t: int) -> np.ndarray:  # (37,) の確率
        raise NotImplementedError


class Uniform(Model):
    name = "ランダム(基準)"

    def predict(self, F, X, t):
        return np.full(N_NUMBERS, BASE_RATE)


class LongRunFrequency(Model):
    name = "通算出現頻度"

    def predict(self, F, X, t):
        return (X[:t].sum(0) + 1) / (t + N_NUMBERS / N_PICK)


class HotNumbers(Model):
    name = "直近20回のホット数字"

    def predict(self, F, X, t):
        return (X[max(0, t - 20) : t].sum(0) + 1) / (min(t, 20) + N_NUMBERS / N_PICK)


class _Tabular(Model):
    retrain_every = 25
    min_history = 100

    def _make(self):
        raise NotImplementedError

    def fit(self, F, X, t_end):
        rows = np.arange(self.min_history, t_end)
        Xtr = F[rows].reshape(-1, F.shape[-1])
        ytr = X[rows].reshape(-1).astype(float)
        self.clf = self._make().fit(Xtr, ytr)

    def predict(self, F, X, t):
        return self.clf.predict_proba(F[t])


class Logistic(_Tabular):
    name = "ロジスティック回帰"

    def _make(self):
        return LogisticRegression(l2=10.0)


class NeuralNet(_Tabular):
    name = "ニューラルネット"
    retrain_every = 50

    def _make(self):
        return MLP(task="binary", hidden=32, epochs=15)


class LSTMModel(Model):
    """直近 L 回の当せんパターン (37 次元 x L) を LSTM に入れ、次回の 37 数字の出現確率を出す。"""

    name = "LSTM(深層学習)"
    retrain_every = 50
    seq_len = 20
    epochs = 40

    def fit(self, F, X, t_end):
        import torch
        from torch import nn

        torch.manual_seed(0)
        L = self.seq_len
        seqs = np.stack([X[t - L : t] for t in range(L, t_end)]).astype(np.float32)
        ys = X[L:t_end].astype(np.float32)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(N_NUMBERS, 64, batch_first=True)
                self.drop = nn.Dropout(0.3)
                self.head = nn.Linear(64, N_NUMBERS)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1]))

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        xs, yt = torch.from_numpy(seqs), torch.from_numpy(ys)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(xs))
            for i in range(0, len(xs), 32):
                idx = perm[i : i + 32]
                opt.zero_grad()
                loss_fn(net(xs[idx]), yt[idx]).backward()
                opt.step()
        net.eval()
        self.net = net

    def predict(self, F, X, t):
        import torch

        with torch.no_grad():
            x = torch.from_numpy(X[t - self.seq_len : t].astype(np.float32))[None]
            return torch.sigmoid(self.net(x))[0].numpy()


ALL_MODELS = [Uniform, LongRunFrequency, HotNumbers, Logistic, NeuralNet, LSTMModel]


# ---------------------------------------------------------------- 評価

@dataclass
class Result:
    name: str
    hits: np.ndarray  # 各テスト回の一致数 (0〜7)
    logloss: float
    seconds: float


def _binary_logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run(models=None, test_start: int = 300, verbose: bool = True) -> list[Result]:
    df = load()
    X = indicator_matrix(df)
    F = number_features(X)
    T = len(X)
    models = models or ALL_MODELS
    results = []
    for cls in models:
        m = cls()
        rng = np.random.default_rng(0)
        t0 = time.time()
        hits, probs = [], []
        for t in range(test_start, T):
            if (t - test_start) % m.retrain_every == 0:
                m.fit(F, X, t)
            p = m.predict(F, X, t)
            # 同率はランダムに崩す (ランダム基準モデルが毎回 1〜7 を選ばないように)
            pick = np.lexsort((rng.random(N_NUMBERS), -p))[:N_PICK]
            hits.append(int(X[t, pick].sum()))
            probs.append(p)
        ll = _binary_logloss(np.array(probs), X[test_start:T])
        r = Result(m.name, np.array(hits), ll, time.time() - t0)
        results.append(r)
        if verbose:
            print(f"  {r.name:<16} 完了 ({r.seconds:.1f}秒)")
    return results


def hypergeom_null():
    """ランダムに 7 個選んだときの一致数の分布。"""
    return stats.hypergeom(N_NUMBERS, N_PICK, N_PICK)


def summarize(results: list[Result]) -> str:
    null = hypergeom_null()
    mu, sd = null.mean(), null.std()
    base_ll = next(r.logloss for r in results if r.name.startswith("ランダム"))
    n = len(results[0].hits)
    lines = [
        f"テスト期間: 直近 {n} 回 (各回、その回より前のデータだけで学習・予測)",
        f"ランダムに 7 個選んだときの平均一致数(理論値): {mu:.3f}  "
        f"(この回数だと偶然でも {mu - 1.96 * sd / np.sqrt(n):.3f}〜{mu + 1.96 * sd / np.sqrt(n):.3f} の範囲に 95% 入る)",
        "",
        f"{'モデル':<18}{'平均一致数':>10}{'p値':>8}{'4個以上一致':>12}{'logloss改善':>12}",
    ]
    for r in results:
        z = (r.hits.mean() - mu) / (sd / np.sqrt(n))
        p = 2 * stats.norm.sf(abs(z))
        ge4 = (r.hits >= 4).sum()
        lines.append(
            f"{r.name:<18}{r.hits.mean():>10.3f}{p:>8.2f}{ge4:>8d}回"
            f"{(base_ll - r.logloss) / base_ll * 100:>+11.2f}%"
        )
    lines += [
        "",
        f"参考: ランダムで 4 個以上一致する回数の期待値 = {n * null.sf(3):.1f}回",
        "p値 < 0.05 でも、モデルを複数試しているので偶然の可能性がある(多重比較)。",
        "logloss改善がプラス = 確率予測がランダムより良い。マイナス = 悪い。",
    ]
    return "\n".join(lines)


def randomness_check() -> str:
    """全期間の数字別出現回数が一様かのカイ二乗検定。"""
    df = load()
    counts = indicator_matrix(df).sum(0)
    chi2, p = stats.chisquare(counts)
    top = [int(j) for j in np.argsort(-counts)[:5] + 1]
    bottom = [int(j) for j in np.argsort(counts)[:5] + 1]
    return (
        f"出現回数のカイ二乗検定 ({len(df)}回分): χ²={chi2:.1f}, p={p:.3f} "
        f"({'偏りあり' if p < 0.05 else '偏りがあるとは言えない'})\n"
        f"  最多: {top}  最少: {bottom}  "
        f"(期待値 {len(df) * BASE_RATE:.1f}回、実際 {counts.min()}〜{counts.max()}回)"
    )


def plot(results: list[Result], path=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    mu = hypergeom_null().mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        if r.name.startswith("ランダム"):
            continue
        ax.plot(np.cumsum(r.hits - mu), label=r.name, lw=1.5)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("テスト回数")
    ax.set_ylabel("ランダムに対する累積の超過一致数")
    ax.set_title("予測モデル vs ランダム(0 より上ならランダムより多く当てている)")
    ax.legend()
    fig.tight_layout()
    path = path or REPORT_DIR / "backtest.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"グラフを保存: {path}")


def main(test_start: int = 300) -> None:
    print(randomness_check())
    print("\nバックテスト実行中...")
    results = run(test_start=test_start)
    print()
    print(summarize(results))
    plot(results)
