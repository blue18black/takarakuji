import argparse
import sys


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(prog="python -m loto7", description="ロト7 データ分析ツール")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="過去の当せんデータを取得して data/loto7.csv に保存")
    bt = sub.add_parser("backtest", help="予測モデル vs ランダムのバックテスト")
    bt.add_argument("--test-start", type=int, default=300, help="この回数目以降をテスト期間にする (既定: 300)")
    sub.add_parser("popularity", help="人気度モデルを学習・評価")
    pk = sub.add_parser("pick", help="被りにくい組み合わせを生成")
    pk.add_argument("-n", type=int, default=5, help="口数 (既定: 5)")
    pk.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.cmd == "fetch":
        from .data import fetch
        fetch()
    elif args.cmd == "backtest":
        from .backtest import main as run
        run(test_start=args.test_start)
    elif args.cmd == "popularity":
        from .popularity import main as run
        run()
    elif args.cmd == "pick":
        from .picker import main as run
        run(args.n, args.seed)


if __name__ == "__main__":
    main()
