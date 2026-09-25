# DSA05：异步周期业务下的上行子信道与功率分布式控制

中心 AP 的上行场景：N 个 UE 共享 K 个子信道，周期性产生小包，各 UE 相位异步，截止期为本 UE 的下一次到达。每个时隙，每个 UE 只根据本地因果信息选择（子信道，功率）或 OFF。

- **速率**：由 Shannon 公式给出（HARQ-IR 累积互信息），MCS 不再是动作。
- **目标**：在每 UE 违约率 ≤ ε 的约束下，最小化发射能耗。

规范见 [场景 v1.0](docs/research/scenario.md)。**场景参数与建模假设已于 2026-09-24 冻结**：N=4/6/8/12 四个变体（K=4，功率档 OFF/5/10/15/19/23 dBm，21 类动作），见 [配置](configs/) 与 [冻结清单](docs/research/scenario-v1-freeze.json)；约束式 MAPPO（[EXP-DSA05-001](docs/experiment-designs/001-constrained-mappo_20260924.md)）已实现，第 1 阶段（N=8 短训）已完成，见结果表。所有 MARL 实验遵守 [MARL 实验协议 v1](docs/research/marl-protocol-v1.md)：训练、验证、测试实例划分与执行种子已冻结。

## 结果

| 报告 | 内容 |
|---|---|
| [N=4 中央最小能耗参照](docs/results/central-reference-n4_20260924/report.md) | 精确最优 3.70 mW/UE，100 实例环境重放核验；本地启发式能耗为最优的 5.5–11 倍 |
| [N=4/6/8/12 中央最小能耗上下界](docs/results/central-bounds_20260924/report.md) | N=4 全部精确、N=6 上界/下界均值 1.02、N=8 为 1.12、N=12 为 1.42；启发式能耗为上界的 8–10 倍 |
| [协议 v1 验证集与测试集参照](docs/results/protocol-v1-references_20260924/report.md) | 四个 N 逐实例上下界（验证 64、测试 200），作为 MARL 的 E/LB、E/UB 分母 |
| [本地基线验证集校准](docs/results/baseline-calibration_20260924/report.md) | 最强可行基线 paced_sticky：N=4 已最优，N=6/8/12 能耗/上界为 1.78/2.87/5.50 |
| [EXP-DSA05-001 第 1 阶段](docs/results/exp001-phase1_20260925/report.md) | N=8 短训：固定惩罚与逐 UE 对偶满足约束，能耗/上界 2.34/2.40（基线 2.87）；全局拉格朗日放弃长周期远端 UE 而失败 |
| [能耗损失诊断](docs/results/exp001-efficiency-diagnosis_20260925/report.md) | 学到的策略约 70% 的发送与他人撞在同一子信道、损失约 40% 互信息；中央最优靠分时完全错开 |

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

## 约束式 MAPPO 训练

```bash
uv run --locked python -m dsa_period.experiments.train_constrained_mappo \
  --scenario configs/scenario-v1-n8.toml --variant local --seed 101 \
  --output outputs/exp001-n8-local-seed101 --formal
```

`--variant` 为 `fixed`（固定惩罚）、`global`（全局拉格朗日）或 `local`（逐 UE 局部对偶）。每次验证都保存检查点，结束时按协议规则选模并写入 `result.json`。`--budget-slots`、`--validation-instances`、`--set` 只用于非正式的冒烟和诊断，与 `--formal` 互斥。

## 目录

```text
configs/scenario-v1-n{4,6,8,12}.toml  # 冻结的 v1.0 工作点（勿改，变更另立新版本）
configs/exp001-constrained-mappo.toml  # EXP-DSA05-001 训练与约束参数
scripts/analyze_power_menu_quantization.py  # 功率档选型分析
scripts/summarize_central_*.py  # 中央参照结果汇总
scripts/summarize_policy_evaluation.py  # 协议评估结果汇总
scripts/explore_mac_sum_rate_bound.py  # MAC 和速率下界探索（负面结果）
src/dsa_period/
  topology.py    # 圆盘拓扑与路损（复制自 dsa04）
  channel.py     # ChannelModel 接口与 StaticChannel；衰落模型在此扩展
  physics.py     # Shannon/TIN 内核，动作为（子信道，功率）
  traffic.py     # 异构周期、异步相位的到达
  env.py         # 连续运行的环境：完成、截止期结算、逾期丢弃、新到达
  metrics.py     # 固定窗口 OCR/能耗统计
  baselines.py   # 本地启发式与本地链路估计
  policies.py    # 统一策略接口；强本地基线 paced_sticky（记住增益、粘住子信道、按节奏选功率）
  evaluation.py  # 协议 v1 评估：逐回合可行判定、E/UB 归一化、两层 bootstrap、选模规则
  config.py      # TOML 加载与实例生成
  splits.py      # 协议 v1 冻结的实例划分与评估执行种子
  central.py     # 逐 UE 单独最优：N<=K 时精确，任意 N 下为下界
  central_timeshare.py  # 正交分时循环调度 MILP：任意 N 下的上界
  experiments/scenario_smoke.py
  experiments/central_reference.py
  experiments/central_bounds.py
  experiments/evaluate_policies.py  # 按协议在 dev/validation/test 上评估本地策略
  experiments/train_constrained_mappo.py  # EXP-DSA05-001 单次训练（场景、约束变体、种子）
  marl/features.py  # Actor 本地特征、截止期结算、逐 UE 对偶变量 μ_i
  marl/networks.py  # CSG Actor（逐子信道共享 GRU）与集中式 Critic
  marl/ppo.py       # 循环 MAPPO，折扣 GAE，截断 bootstrap
  marl/policy.py    # 训练好的 Actor 的协议评估包装（逐 UE RNG 随机执行）
  marl/trainer.py   # 采样、奖励与代价、验证、检查点、协议选模
docs/research/scenario.md       # 场景规范
docs/research/scenario-v1-freeze.json  # 冻结清单：配置 SHA-256、确认的假设、派生量
docs/research/marl-protocol-v1.md   # MARL 实验协议：划分、选模、测试、指标与判读
docs/research/marl-protocol-v1.json # 协议清单：划分区间、执行种子、实例哈希
tests/
```

核心接口为 `compute_step(config, menu, gains, actions)` 和 `PeriodicUplinkEnv(config, menu, channel, traffic, horizon_slots)`。环境不计算奖励，只返回逐 UE 的能耗和截止期结算结果，奖励与约束代价由训练器构造。
