"""人気度モデルを使って「当たったときに他の人と被りにくい」組み合わせを選ぶ。

当せん確率はどの組み合わせでも同じ (1等 1/10,295,472)。
ここで上げられるのは「当たったときの1口あたりの取り分」だけ。
"""

from __future__ import annotations

import numpy as np

from .data import MAIN_COLS, N_NUMBERS, load
from .popularity import combo_features, load_model, training_data


def random_combos(n: int, rng: np.random.Generator) -> np.ndarray:
    keys = rng.random((n, N_NUMBERS))
    return np.sort(np.argpartition(keys, 7, axis=1)[:, :7] + 1, axis=1)


def pick(n_tickets: int = 5, n_candidates: int = 300_000, max_shared: int = 2, seed: int | None = None):
    df = load()
    model = load_model()
    prev = df[MAIN_COLS].to_numpy()[-1]
    past = {tuple(sorted(r)) for r in df[MAIN_COLS].to_numpy()}

    rng = np.random.default_rng(seed)
    cands = random_combos(n_candidates, rng)
    feats = combo_features(cands, prev)
    score = model.predict(feats)

    # 外挿の防止: 過去の当せん番号で見たことのある範囲を超える特徴を持つ組み合わせは除外する。
    # (例: 連番が5つ並ぶような組み合わせは学習データにほぼ無く、人気度の推定が当てにならない)
    _, Xtr, _, _ = training_data()
    n_shape = Xtr.shape[1] - N_NUMBERS  # 数字ごとの one-hot 以外の特徴
    lo, hi = np.percentile(Xtr[:, :n_shape], [1, 99], axis=0)
    in_range = ((feats[:, :n_shape] >= lo) & (feats[:, :n_shape] <= hi)).all(1)
    # 「前回と重複が多いほど不人気」は単独の相関がほぼゼロで根拠が弱い。
    # 前回の当せん番号を買う人は一定数いるはずなので、重複は 2 個までに抑える。
    in_range &= np.isin(cands, prev).sum(1) <= 2
    order = np.argsort(-np.where(in_range, score, -np.inf))

    chosen: list[int] = []
    for i in order:
        c = set(cands[i])
        if tuple(cands[i]) in past:
            continue  # 過去の当せん番号をそのまま買う人は多い
        if all(len(c & set(cands[j])) <= max_shared for j in chosen):
            chosen.append(i)
        if len(chosen) == n_tickets:
            break

    worst = np.argmin(np.where(in_range, score, np.inf))
    return {
        "tickets": cands[chosen],
        "scores": score[chosen],
        "median_score": float(np.median(score)),
        "worst": (cands[worst], float(score[worst])),
        "latest": df.iloc[-1],
    }


def main(n_tickets: int = 5, seed: int | None = None) -> None:
    r = pick(n_tickets, seed=seed)
    latest = r["latest"]
    print(f"最新データ: 第{latest['draw']}回 ({latest['date']:%Y-%m-%d})  "
          f"本数字 {' '.join(str(latest[c]) for c in MAIN_COLS)}  "
          f"キャリーオーバー {latest['carryover']:,}円")
    print(f"\n被りにくい組み合わせ {len(r['tickets'])} 口 (各口の共通数字は2個まで):")
    for t, s in zip(r["tickets"], r["scores"]):
        rel = np.exp(s - r["median_score"]) - 1
        print(f"  {' '.join(f'{x:2d}' for x in t)}   当たったときの取り分: 平均的な組み合わせより約 {rel:+.0%}")
    w, ws = r["worst"]
    print(f"\n参考: 一番人気だと推定された組み合わせ {' '.join(map(str, w))} "
          f"(平均より約 {np.exp(ws - r['median_score']) - 1:+.0%})")
    print("\n※ 当せん確率はどの組み合わせでも同じです。変わるのは山分けになったときの取り分の期待値だけです。")
    print("※ 取り分の % は 3〜5等の当せん口数から学習した推定値です。1等での効果はこれより大きくも小さくもなり得ます。")
