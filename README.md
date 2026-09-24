# AlphaGPT · A股截面因子挖掘

原 [imbue-bit/AlphaGPT](https://github.com/imbue-bit/AlphaGPT)（Solana meme 币
"RL 生成因子公式 + 链上执行"系统）的 fork，已改造为 **A 股全市场主板截面
因子挖掘** 项目（定位：因子合成器）。原加密交易链路已移除（见 git 历史），
仅保留了被验证过的方法论骨架：**DSL 词表 + 栈式虚拟机 + Transformer 生成器
+ REINFORCE**。

## 核心思路

```
Transformer 采样公式 token 序列（6 特征原子 + 12 算子，≤12 长）
    → StackVM 在 [股票 N, 网格日 T] 面板上解释执行，产出截面分数
    → 分数过中性化链（MAD → z → 申万行业+log市值 → 残差 z）
    → 与 t8 池内超额做逐网格日 Spearman RankIC → reward
    → REINFORCE 更新生成器；top-K 按样本外（2024+）ICIR 裁决
```

训练与评估共用同一份 `StackVM.execute`——回测语义即执行语义。

## 红线约束

唯一外部代码依赖是 [GMtest](../GMtest) 仓库的 `utils/` 数据层（掘金 SDK 的
缓存封装），经 `ashare/gm_bridge.py` 唯一转引点导入；其余一切逻辑
（宇宙过滤 / t8 标签 / 中性化 / 评估）在本仓库本地实现，课题契约按
GMtest cross_section 课题的文档约定复刻。`tests/test_redline.py` 源码扫描
强制此约定。

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
set GMTEST_HOME=D:\workspace\GMtest     # 数据源仓库（可选，默认即此路径）

python -m ashare.run_mining materialize        # GMtest 缓存 → ashare/cache/panel.npz
python -m ashare.run_mining train --steps 50   # 短跑验证；完整训练默认 1000 步
python -m ashare.run_mining inspect            # 面板/产物概况
python -m ashare.scripts.validate_vs_gmtest    # 与课题采样池缓存对账（只读）
pytest ashare/tests -q                         # 19 项测试
```

数据物化读 GMtest 磁盘缓存（日线/快照/估值/行业/市值），缺失段经 gm SDK
网络补拉，不依赖掘金终端。

## 目录

```
ashare/
├── gm_bridge.py        # GMtest utils 唯一转引点（红线执行处）
├── grid/universe/labels.py   # 课题契约复刻：8td 网格 / 可交易池 / t8 复权标签
├── features.py         # 6 原子：REV5/BIAS60/TURNOVER/VRATIO/VOL20/EP
├── materialize.py      # 面板物化（训练/评估唯一数据入口）
├── ops/vocab/vm.py     # 12 算子 + 词表 + 栈机（自原 model_core 移植）
├── preprocess.py       # 中性化链（Phase 2）
├── evaluator.py        # RankIC / ICIR / 分层（与 scipy.spearmanr 对账）
├── alphagpt/engine.py  # 生成器 + REINFORCE（含 LoRD 正则）
└── tests/              # 红线 / VM / 评估器 / 中性化 / 数据试金石
```

阶段状态与边界见 [ashare/README.md](ashare/README.md)。

## 许可

Apache-2.0（沿袭上游仓库；`ops/vocab/vm/alphagpt/engine` 含自上游移植的代码）。
