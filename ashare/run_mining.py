# -*- coding: utf-8 -*-
"""A股截面因子挖掘 CLI。

用法（项目环境）：
  python -m ashare.run_mining materialize          # 物化面板（GMtest 缓存 → npz）
  python -m ashare.run_mining train [--steps 50]   # REINFORCE 训练（默认 1000 步）
  python -m ashare.run_mining inspect              # 查看面板/公式产物概况
"""
import argparse
import json

from .config import OUTPUT_DIR, PANEL_PATH


def cmd_materialize(args):
    from .materialize import materialize
    meta = materialize(end_date=args.end)
    print(json.dumps({k: v for k, v in meta.items() if k != "grid"},
                     ensure_ascii=False, indent=2))


def cmd_train(args):
    from .config import TRAIN_STEPS
    from .engine import AlphaEngine
    eng = AlphaEngine(use_lord_regularization=not args.no_lord)
    eng.train(steps=args.steps if args.steps is not None else TRAIN_STEPS)


def cmd_inspect(_args):
    import numpy as np
    if not PANEL_PATH.exists():
        print(f"面板不存在: {PANEL_PATH}（先 materialize）")
        return
    data = np.load(PANEL_PATH, allow_pickle=True)
    mask, labels = data["mask"], data["labels"]
    print(f"股票数 {len(data['symbols'])} | 采样日 {len(data['grid_dates'])} "
          f"({data['grid_dates'][0]} ~ {data['grid_dates'][-1]})")
    print(f"池均 {mask.sum() / mask.shape[1]:.0f} 只 | "
          f"标签有效率 {np.isfinite(labels[mask]).mean():.1%}")
    for name in ("formula_candidates.json", "training_history.json"):
        p = OUTPUT_DIR / name
        if p.exists():
            print(f"\n== {name} ==")
            print(p.read_text(encoding="utf-8")[:1200])


def main():
    parser = argparse.ArgumentParser(description="A股截面因子挖掘（定位 B）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mat = sub.add_parser("materialize", help="物化面板")
    p_mat.add_argument("--end", default=None, help="数据窗口右端（默认安全末收盘日）")
    p_mat.set_defaults(func=cmd_materialize)

    p_train = sub.add_parser("train", help="REINFORCE 训练")
    p_train.add_argument("--steps", type=int, default=None)
    p_train.add_argument("--no-lord", action="store_true", help="关闭 LoRD 正则")
    p_train.set_defaults(func=cmd_train)

    p_ins = sub.add_parser("inspect", help="查看面板/产物")
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
