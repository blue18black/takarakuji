"""ロト7の過去データ取得・整形。

データ元: Chaos World (個人運営サイト) が公開している JavaScript 配列ファイル。
  - data7_basic.js : 抽せん回, 本数字7個, ボーナス数字2個
  - data7_prize.js : 抽せん日, 1〜6等の1口あたり当せん金
  - data7_share.js : 1〜6等の当せん口数
  - data7_carry.js : キャリーオーバー額

相手サイトへの負荷を避けるため、取得は `fetch` を明示的に実行したときだけ行い、
結果は data/loto7.csv にキャッシュする。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.ne.jp/asahi/chaosworld/loto6/"
FILES = ["data7_basic.js", "data7_prize.js", "data7_share.js", "data7_carry.js"]

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CSV_PATH = DATA_DIR / "loto7.csv"

N_NUMBERS = 37
N_PICK = 7
MAIN_COLS = [f"n{i}" for i in range(1, 8)]
BONUS_COLS = ["b1", "b2"]
TIERS = range(1, 7)


def download_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (loto7-research script)"}
    for name in FILES:
        r = requests.get(BASE_URL + name, headers=headers, timeout=30)
        r.raise_for_status()
        (RAW_DIR / name).write_bytes(r.content)
        print(f"downloaded {name} ({len(r.content):,} bytes)")


def _read_array(name: str) -> list[str]:
    text = (RAW_DIR / name).read_bytes().decode("shift_jis", errors="replace")
    m = re.search(r"new Array\((.*?)\)", text, re.S)
    if not m:
        raise ValueError(f"{name}: 配列が見つかりません")
    return [t.strip().strip('"') for t in m.group(1).split(",") if t.strip()]


def _chunk(values: list[str], size: int, name: str) -> list[list[str]]:
    if len(values) % size:
        raise ValueError(f"{name}: 要素数 {len(values)} が {size} の倍数ではありません")
    return [values[i : i + size] for i in range(0, len(values), size)]


def build_dataframe() -> pd.DataFrame:
    basic = _chunk(_read_array("data7_basic.js"), 10, "basic")
    prize = _chunk(_read_array("data7_prize.js")[1:], 7, "prize")  # 先頭は "dummy"
    share = _chunk(_read_array("data7_share.js"), 6, "share")
    carry = _read_array("data7_carry.js")  # その回の抽せん後に次回へ繰り越された額

    n = len(basic)
    for label, arr in [("prize", prize), ("share", share), ("carry", carry)]:
        if len(arr) != n:
            raise ValueError(f"{label} の回数 {len(arr)} が当せん番号の回数 {n} と一致しません")

    rows = []
    for b, p, s, c in zip(basic, prize, share, carry):
        row = {"draw": int(b[0]), "date": p[0]}
        row.update({col: int(v) for col, v in zip(MAIN_COLS + BONUS_COLS, b[1:])})
        row.update({f"prize{t}": int(v) for t, v in zip(TIERS, p[1:])})
        row.update({f"winners{t}": int(v) for t, v in zip(TIERS, s)})
        row["carryover"] = int(c)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    assert (df["draw"].diff().dropna() == 1).all(), "抽せん回が連番ではありません"
    nums = df[MAIN_COLS + BONUS_COLS].to_numpy()
    assert ((nums >= 1) & (nums <= N_NUMBERS)).all(), "1〜37 以外の数字があります"
    for i, row in enumerate(nums):
        assert len(set(row)) == 9, f"第{df['draw'].iat[i]}回: 数字が重複しています"
    assert df["date"].is_monotonic_increasing, "日付が昇順ではありません"
    # 当せん者がいる等級は賞金 > 0、いない等級は 0 であること
    for t in TIERS:
        has = df[f"winners{t}"] > 0
        assert (df.loc[has, f"prize{t}"] > 0).all(), f"{t}等: 当せん者がいるのに賞金が 0 の回があります"


def fetch() -> pd.DataFrame:
    download_raw()
    df = build_dataframe()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"saved {CSV_PATH} : 第{df['draw'].min()}回〜第{df['draw'].max()}回 ({len(df)}回分, 最新 {df['date'].max():%Y-%m-%d})")
    return df


def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError("data/loto7.csv がありません。先に `python -m loto7 fetch` を実行してください。")
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


def main_numbers(df: pd.DataFrame):
    """(回数, 7) の本数字配列。"""
    return df[MAIN_COLS].to_numpy()


def indicator_matrix(df: pd.DataFrame):
    """(回数, 37) の 0/1 行列。列 j は数字 j+1 が本数字に入ったかどうか。"""
    import numpy as np

    nums = main_numbers(df)
    X = np.zeros((len(df), N_NUMBERS), dtype=np.int8)
    rows = np.repeat(np.arange(len(df)), N_PICK)
    X[rows, nums.ravel() - 1] = 1
    return X
