"""組み合わせの「人気度」を、過去の当せん口数から学習する。

考え方
------
2〜6等はパリミュチュエル方式(その等級の賞金総額を当せん者で山分け)。
その等級の賞金総額は売上に比例するので、1口あたり当せん金は

    当せん金_k ∝ 売上 / 当せん口数_k ∝ 1 / (確率_k × その回の当せん番号の人気度)

となり、売上の影響が打ち消される。つまり
「1口あたりの当せん金が普段より高い回 = 当せん番号が不人気だった回」。

抽せんは完全ランダムなので、どの回にどんな番号が出たかは「人気度とは無関係に
割り当てられた実験条件」になっている。だから当せん番号の特徴 → 当せん金の
高さ、という回帰は因果的に解釈しやすい(交絡がない)。

目的変数: 3・4・5等の log(当せん金) から、前後の回の中央値を引いた残差の平均。
  (ルール改定による賞金配分の変化を前後の中央値で吸収する)
  + なら不人気(当たったら取り分が多い)、- なら人気。
  1等は上限・キャリーオーバーの影響が大きく、2等は口数が少なすぎるので使わない。
  6等はボーナス数字が絡むので使わない。

注意: 3〜5等は「7個中 6/5/4 個一致」した人の口数なので、厳密には
「その部分集合を含む組み合わせの人気」を測っている。1等で山分けになるかは
7個全部一致の人気だが、両者は強く相関すると仮定している。
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from .data import DATA_DIR, MAIN_COLS, N_NUMBERS, load
from .ml import MLP, RidgeCV, kfold_indices

TARGET_TIERS = (3, 4, 5)
ROLL_WINDOW = 41
MODEL_PATH = DATA_DIR / "popularity_model.pkl"


# ---------------------------------------------------------------- 目的変数

def unpopularity_target(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    resid = {}
    for k in TARGET_TIERS:
        logp = np.log(df[f"prize{k}"].where(df[f"prize{k}"] > 0))
        trend = logp.rolling(ROLL_WINDOW, center=True, min_periods=10).median()
        resid[k] = logp - trend
    resid = pd.DataFrame(resid)
    return resid.mean(axis=1), resid


# ---------------------------------------------------------------- 特徴量

FEATURE_NAMES: list[str] = []


def combo_features(combos: np.ndarray, prev_draw: np.ndarray | None = None) -> np.ndarray:
    """combos: (N, 7) の本数字。prev_draw: 直前の回の本数字 (N, 7) か (7,)。"""
    combos = np.sort(np.asarray(combos), axis=1)
    N = len(combos)
    onehot = np.zeros((N, N_NUMBERS))
    onehot[np.repeat(np.arange(N), 7), combos.ravel() - 1] = 1

    diffs = np.diff(combos, axis=1)
    last_digit = combos % 10
    same_digit_max = np.array([np.bincount(r, minlength=10).max() for r in last_digit])
    decade = (combos - 1) // 10
    decade_max = np.array([np.bincount(r, minlength=4).max() for r in decade])

    feats = {
        "誕生日範囲(1-31)の個数": (combos <= 31).sum(1),
        "月の範囲(1-12)の個数": (combos <= 12).sum(1),
        "32以上の個数": (combos >= 32).sum(1),
        "合計": combos.sum(1),
        "奇数の個数": (combos % 2).sum(1),
        "連番ペア数": (diffs == 1).sum(1),
        "最大の間隔": diffs.max(1),
        "等間隔の連続": (np.diff(diffs, axis=1) == 0).sum(1),
        "下一桁が同じ数の最大個数": same_digit_max,
        "同じ10の位の最大個数": decade_max,
        "7の倍数の個数": (combos % 7 == 0).sum(1),
        "ゾロ目(11,22,33)の個数": np.isin(combos, [11, 22, 33]).sum(1),
    }
    if prev_draw is not None:
        prev = np.broadcast_to(np.asarray(prev_draw), combos.shape)
        feats["前回当せん番号との重複数"] = np.array([len(set(a) & set(b)) for a, b in zip(combos, prev)])
    else:
        feats["前回当せん番号との重複数"] = np.zeros(N)

    global FEATURE_NAMES
    FEATURE_NAMES = list(feats) + [f"数字{j}" for j in range(1, N_NUMBERS + 1)]
    return np.column_stack([np.column_stack(list(feats.values())), onehot]).astype(float)


def training_data():
    df = load()
    y, resid = unpopularity_target(df)
    combos = df[MAIN_COLS].to_numpy()
    prev = np.vstack([np.zeros((1, 7), dtype=int), combos[:-1]])
    Xf = combo_features(combos, prev)
    ok = y.notna().to_numpy().copy()
    ok[0] = False  # 初回は前回がない
    return df[ok], Xf[ok], y[ok].to_numpy(), resid[ok]


# ---------------------------------------------------------------- 学習・評価

def make_ridge():
    return RidgeCV()


def make_mlp():
    return MLP(task="regression", hidden=32, epochs=150, weight_decay=1e-2)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def evaluate() -> dict:
    _, X, y, resid = training_data()
    out = {"n": len(y), "tier_corr": resid.corr()}
    for name, mk in [("リッジ回帰", make_ridge), ("ニューラルネット", make_mlp)]:
        pred = np.zeros_like(y)
        for tr, va in kfold_indices(len(y), 5, seed=1):  # alpha 選択とは別の分割で評価
            pred[va] = mk().fit(X[tr], y[tr]).predict(X[va])
        out[name] = {"r2": r2(y, pred), "corr": np.corrcoef(y, pred)[0, 1]}
    return out


def train() -> object:
    _, X, y, _ = training_data()
    model = make_ridge().fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model():
    if not MODEL_PATH.exists():
        return train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def coefficients(model) -> pd.Series:
    """標準化した特徴量 1 単位あたりの効果 (+ = 不人気寄り)。"""
    return pd.Series(model.coef, index=FEATURE_NAMES)


def main() -> None:
    ev = evaluate()
    print(f"学習データ: {ev['n']}回分")
    print("\n[信号の確認] 3・4・5等の賞金残差の相関 (人気度という共通要因があれば正の相関になる)")
    print(ev["tier_corr"].round(2).to_string())
    print("\n[交差検証] 当せん番号の特徴から賞金の高さ(=不人気度)をどれだけ当てられるか")
    for name in ("リッジ回帰", "ニューラルネット"):
        print(f"  {name:<12} R²={ev[name]['r2']:.3f}  相関={ev[name]['corr']:.3f}")

    model = train()
    coef = coefficients(model)
    print(f"\n全データで学習したリッジ回帰を保存: {MODEL_PATH}")
    feats = coef[~coef.index.str.startswith("数字")].sort_values()
    nums = coef[coef.index.str.startswith("数字")].sort_values()
    print("\n[組み合わせの特徴の効果]  - = 人気(避けたい)  + = 不人気(狙い目)")
    for k, v in feats.items():
        print(f"  {k:<22}{v:+.4f}")
    print("\n[個別の数字]  人気上位5:", ", ".join(f"{k}({v:+.4f})" for k, v in nums.head(5).items()))
    print("              不人気上位5:", ", ".join(f"{k}({v:+.4f})" for k, v in nums.tail(5)[::-1].items()))
