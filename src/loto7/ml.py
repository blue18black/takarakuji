"""小さな機械学習ユーティリティ (numpy / scipy / PyTorch のみ)。

この環境では Windows のアプリケーション制御ポリシーが scikit-learn の DLL を
ブロックするため、必要な部分だけを自前で実装している。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class Standardizer:
    def fit(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def kfold_indices(n: int, k: int = 5, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate(folds[:i] + folds[i + 1 :]), folds[i]


class LogisticRegression:
    """L2 正則化つきロジスティック回帰 (L-BFGS)。"""

    def __init__(self, l2: float = 10.0):
        self.l2 = l2

    def fit(self, X, y):
        self.sc = Standardizer().fit(X)
        Z = np.column_stack([self.sc.transform(X), np.ones(len(X))])
        n = len(y)

        def loss(w):
            z = Z @ w
            p = 1 / (1 + np.exp(-z))
            nll = np.logaddexp(0, z).sum() - y @ z
            reg = 0.5 * self.l2 * (w[:-1] @ w[:-1])
            grad = Z.T @ (p - y)
            grad[:-1] += self.l2 * w[:-1]
            return (nll + reg) / n, grad / n

        self.w = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict_proba(self, X):
        Z = np.column_stack([self.sc.transform(X), np.ones(len(X))])
        return 1 / (1 + np.exp(-(Z @ self.w)))


class RidgeCV:
    """標準化 + リッジ回帰。正則化の強さ alpha は k-fold 交差検証で選ぶ。"""

    def __init__(self, alphas=np.logspace(-1, 4, 30), k: int = 5):
        self.alphas = alphas
        self.k = k

    @staticmethod
    def _solve(Z, y, alpha):
        ym = y.mean()
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        return np.linalg.solve(A, Z.T @ (y - ym)), ym

    def fit(self, X, y):
        self.sc = Standardizer().fit(X)
        Z = self.sc.transform(X)
        errs = np.zeros(len(self.alphas))
        for tr, va in kfold_indices(len(y), self.k):
            for i, a in enumerate(self.alphas):
                w, b = self._solve(Z[tr], y[tr], a)
                errs[i] += ((Z[va] @ w + b - y[va]) ** 2).sum()
        self.alpha = self.alphas[errs.argmin()]
        self.coef, self.intercept = self._solve(Z, y, self.alpha)
        return self

    def predict(self, X):
        return self.sc.transform(X) @ self.coef + self.intercept


class MLP:
    """PyTorch の 2 層ニューラルネット。task='binary' か 'regression'。"""

    def __init__(self, task: str = "binary", hidden: int = 32, epochs: int = 60, lr: float = 1e-3,
                 weight_decay: float = 1e-3, dropout: float = 0.3, seed: int = 0):
        self.task, self.hidden, self.epochs, self.lr = task, hidden, epochs, lr
        self.weight_decay, self.dropout, self.seed = weight_decay, dropout, seed

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.sc = Standardizer().fit(X)
        xs = torch.tensor(self.sc.transform(X), dtype=torch.float32)
        ys = torch.tensor(y, dtype=torch.float32)
        if self.task == "regression":
            self.ymean, self.ystd = float(ys.mean()), float(ys.std()) or 1.0
            ys = (ys - self.ymean) / self.ystd
        self.net = nn.Sequential(
            nn.Linear(xs.shape[1], self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss() if self.task == "binary" else nn.MSELoss()
        bs = 256 if len(xs) > 5000 else 32
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(xs))
            for i in range(0, len(xs), bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                loss_fn(self.net(xs[idx]).squeeze(1), ys[idx]).backward()
                opt.step()
        self.net.eval()
        return self

    def _raw(self, X):
        import torch

        with torch.no_grad():
            return self.net(torch.tensor(self.sc.transform(X), dtype=torch.float32)).squeeze(1).numpy()

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self._raw(X)))

    def predict(self, X):
        return self._raw(X) * self.ystd + self.ymean
