# DSA05：异步周期业务下的上行子信道与功率分布式控制

中心 AP 的上行场景：N 个 UE 共享 K 个子信道，周期性产生小包，各 UE 相位异步，截止期为本 UE 的下一次到达。每个时隙，每个 UE 只根据本地因果信息选择（子信道，功率）或 OFF。

- **速率**：由 Shannon 公式给出（HARQ-IR 累积互信息），MCS 不再是动作。
- **目标**：在每 UE 违约率 ≤ ε 的约束下，最小化发射能耗。

规范见 [场景 v1.0](docs/research/scenario.md)。**场景参数与建模假设已于 2026-09-24 冻结**：N=4/6/8/12 四个变体（K=4，功率档 OFF/5/10/15/19/23 dBm，21 类动作），见 [配置](configs/) 与 [冻结清单](docs/research/scenario-v1-freeze.json)；MARL 训练尚未接入。所有 MARL 实验遵守 [MARL 实验协议 v1](docs/research/marl-protocol-v1.md)：训练、验证、测试实例划分与执行种子已冻结。

## 结果

| 报告 | 内容 |
|---|---|
| [N=4 中央最小能耗参照](docs/results/central-reference-n4_20260924/report.md) | 精确最优 3.70 mW/UE，100 实例环境重放核验；本地启发式能耗为最优的 5.5–11 倍 |
| [N=4/6/8/12 中央最小能耗上下界](docs/results/central-bounds_20260924/report.md) | N=4 全部精确、N=6 上界/下界均值 1.02、N=8 为 1.12、N=12 为 1.42；启发式能耗为上界的 8–10 倍 |

## 环境

Python 3.12 + uv：

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
```

## 场景烟测与负载扫描

```bash
uv run --locked python -m dsa_period.experiments.scenario_smoke --instances 4   # 默认 N=12
uv run --locked python -m dsa_period.experiments.scenario_smoke --config configs/scenario-v1-n8.toml
uv run --locked python -m dsa_period.experiments.scenario_smoke --instances 6 --packet-scale 4 \
  --output outputs/smoke-scale4
```

在冻结配置下，对若干实例运行三个本地启发式（`random_max`、`sticky_max`、`lazy_pacing`），输出固定窗口内的按期完成率、最差 UE 完成率、每 UE 功率、每个按期包的能耗和时延。`--packet-scale` 按比例放大所有包长，用于调节负载。输出目录已存在时，程序会拒绝写入。结果标记为 `formal=false`，只作为集成检查。

## 目录

```text
configs/scenario-v1-n{4,6,8,12}.toml  # 冻结的 v1.0 工作点（勿改，变更另立新版本）
scripts/analyze_power_menu_quantization.py  # 功率档选型分析
scripts/summarize_central_*.py  # 中央参照结果汇总
scripts/explore_mac_sum_rate_bound.py  # MAC 和速率下界探索（负面结果）
src/dsa_period/
  topology.py    # 圆盘拓扑与路损（复制自 dsa04）
  channel.py     # ChannelModel 接口与 StaticChannel；衰落模型在此扩展
  physics.py     # Shannon/TIN 内核，动作为（子信道，功率）
  traffic.py     # 异构周期、异步相位的到达
  env.py         # 连续运行的环境：完成、截止期结算、逾期丢弃、新到达
  metrics.py     # 固定窗口 OCR/能耗统计
  baselines.py   # 本地启发式与本地链路估计
  config.py      # TOML 加载与实例生成
  splits.py      # 协议 v1 冻结的实例划分与评估执行种子
  central.py     # 逐 UE 单独最优：N<=K 时精确，任意 N 下为下界
  central_timeshare.py  # 正交分时循环调度 MILP：任意 N 下的上界
  experiments/scenario_smoke.py
  experiments/central_reference.py
  experiments/central_bounds.py
docs/research/scenario.md       # 场景规范
docs/research/scenario-v1-freeze.json  # 冻结清单：配置 SHA-256、确认的假设、派生量
docs/research/marl-protocol-v1.md   # MARL 实验协议：划分、选模、测试、指标与判读
docs/research/marl-protocol-v1.json # 协议清单：划分区间、执行种子、实例哈希
tests/
```

核心接口为 `compute_step(config, menu, gains, actions)` 和 `PeriodicUplinkEnv(config, menu, channel, traffic, horizon_slots)`。环境不计算奖励，只返回逐 UE 的能耗和截止期结算结果，奖励与约束代价由训练器构造。
