# ashare/ —— A股截面因子挖掘（Phase 1，定位 B：因子合成器）

把 AlphaGPT 的"RL 生成因子公式"范式移植到 A 股全市场主板截面课题：
Transformer 采样公式 token 序列 → StackVM 在 [N 股票, T 网格日] 面板上执行 →
网格日截面 RankIC（对 t8 池内超额）作为 reward → REINFORCE 更新生成器。

## 红线约束

唯一外部代码依赖是 GMtest 的 `utils/` 数据层（`gm_bridge.py` 唯一转引点，
`GMTEST_HOME` 默认 `D:\workspace\GMtest`）。宇宙过滤 / t8 标签 / 预处理契约
按 GMtest cross_section 课题的文档约定**本地复刻**，不 import 其 models 代码。
`tests/test_redline.py` 源码扫描强制此约定。

## 用法（项目环境 `.venv`）

```bash
python -m ashare.run_mining materialize        # GMtest 缓存 → ashare/cache/panel.npz
python -m ashare.run_mining train --steps 50   # 短跑验证；完整训练默认 1000 步
python -m ashare.run_mining inspect            # 面板/产物概况
python -m ashare.scripts.validate_vs_gmtest    # 与课题采样池缓存对账（只读）
pytest ashare/tests -v                         # 单测（红线/VM/评估器/数据试金石）
```

依赖：torch / pandas / pyarrow / gm / scipy / tqdm / pytest（见仓库根安装）。
数据物化读 GMtest 磁盘缓存（日线/快照/估值/行业/市值），缺失段自动经 gm SDK
网络补拉（`GM_TOKEN` 兜底），不依赖掘金终端。

## 模块一览

| 模块 | 职责 |
|---|---|
| `gm_bridge.py` | GMtest utils 唯一转引点（红线执行处） |
| `grid.py` / `universe.py` / `labels.py` | 课题契约本地复刻：8td 网格 / 可交易池 / t8 复权标签 |
| `features.py` | 6 原子（REV5/BIAS60/TURNOVER/VRATIO/VOL20/EP）+ 截面 robust-z |
| `materialize.py` | 面板物化（训练/评估唯一数据入口） |
| `ops.py` / `vocab.py` / `vm.py` | 12 算子 + 词表 + 栈机（自 model_core 移植） |
| `evaluator.py` | 截面 RankIC reward（torch 平均秩，与 scipy.spearmanr 对账） |
| `alphagpt.py` / `engine.py` | 生成器 + REINFORCE（含 LoRD 正则，自 model_core 移植） |

## Phase 边界（本期不做）

中性化链进 reward、ICIR/分层/成本、时间外切分（Phase 2）；
因子池注册与 (symbol,date)→value 导出（Phase 3）。
