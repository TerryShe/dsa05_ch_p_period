---
doc_id: RES-DSA05-BASELINE-CALIBRATION
type: result-report
status: formal
date: 2026-09-24
protocol: marl-protocol-v1
---

# 本地基线在验证集上的校准

按 [MARL 实验协议 v1](../../research/marl-protocol-v1.md)，在验证集（64 个实例 × 1 次执行，冻结执行种子）上评估四个本地基线，并用协议的选模规则为每个 N 选出最强的可行基线。选中的基线是 [EXP-DSA05-001](../../experiment-designs/001-constrained-mappo_20260924.md) 中 MARL 必须超越的对照。本轮不涉及测试集。

## 结论

| N | 选中的基线 | 可行率 F [95% 区间] | 能耗/上界 [95% 区间] | 能耗/下界 | mW/UE |
|---|---|---|---|---|---|
| 4 | paced_sticky（margin=0） | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 3.71 |
| 6 | paced_sticky（margin=0.25） | 0.984 [0.953, 1.000] | 1.782 [1.742, 1.826] | 1.819 | 6.78 |
| 8 | paced_sticky（margin=0.5） | 0.953 [0.891, 1.000] | 2.871 [2.775, 2.973] | 3.207 | 12.18 |
| 12 | paced_sticky（margin=8） | 0.984 [0.953, 1.000] | 5.498 [5.214, 5.785] | 7.865 | 30.08 |

- **N=4 时，本地规则已达到最优**：paced_sticky（margin=0）在全部 64 个实例上都可行，能耗与精确最优相等（逐实例相对误差 < 1e-9）。没有资源竞争时，记住自身增益、按节奏选最低够用的功率就是最优策略，N=4 上 MARL 没有提升空间。
- **N 越大，最强本地基线离最优越远**：能耗/上界从 1.78（N=6）升到 2.87（N=8）和 5.50（N=12）。MARL 的价值要在 N≥6，尤其是 N=8 和 N=12 上证明。
- **在 N=12，margin 与能耗不是单调关系**：margin 太小会违约，而且同时发送的 UE 多、干扰大，反而费能；margin 太大则功率偏高。最省的可行点是 margin=8，比 random_max（能耗/上界 8.74）省约 37%。
- **原有的三个启发式都远不如 paced_sticky**：random_max 和 sticky_max 在各个 N 下的能耗是上界的 8.7–10.5 倍。lazy_pacing 在所有 N 下都不满足约束（F ≤ 0.91）。

## 1. 设置

| 项目 | 取值 |
|---|---|
| 场景 | `configs/scenario-v1-n{4,6,8,12}.toml`（冻结 v1.0） |
| 实例 | 验证集 10,000–10,063，每个实例 1 次执行，使用协议冻结的执行种子 |
| 基线 | random_max、sticky_max、lazy_pacing；paced_sticky 的 margin 取 {0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32} |
| 选模规则 | 在 F ≥ 0.95 的候选中取能耗/上界的均值最小者（协议 §3） |
| 区间 | 两层 bootstrap（这里只有一个"训练种子"，实际只对实例重抽样），10,000 次，种子 20260926 |
| 运行 | `--formal`，提交 `da7a482`，工作区干净 |
| 原始输出 | `outputs/baseline-calibration-validation-n{N}_20260924/summary.json`（Git 忽略），SHA-256 记录在 [summary.json](summary.json) |

paced_sticky 的 margin 网格最初只到 1。在开发池（实例 100–115）上试探时发现，N=12 时 margin 取 1 仍全部不满足约束，而更大的 margin 反而既可行又更省能，所以正式校准前把网格扩到 32。这次试探没有用到验证集或测试集。

## 2. 全部结果

能耗/上界与 F 取均值。"满足"指 F ≥ 0.95。

| 基线 | N=4 F / 能耗/上界 | N=6 | N=8 | N=12 |
|---|---|---|---|---|
| random_max | 1.000 / 9.78 | 1.000 / 9.48 | 1.000 / 9.36 | 0.984 / 8.74 |
| sticky_max | 1.000 / 10.30 | 1.000 / 10.46 | 1.000 / 10.49 | 0.984 / 10.08 |
| lazy_pacing | 0.906 / 5.25 | 0.766 / 5.00 | 0.250 / 5.87 | 0.000 / 8.05 |
| paced_sticky 0 | **1.000 / 1.000** | 0.547 / 1.733 | 0.016 / 2.687 | 0.000 / 8.748 |
| paced_sticky 0.1 | 1.000 / 1.015 | 0.875 / 1.748 | 0.141 / 2.745 | 0.000 / 8.875 |
| paced_sticky 0.25 | 1.000 / 1.043 | **0.984 / 1.782** | 0.406 / 2.778 | 0.000 / 9.023 |
| paced_sticky 0.5 | 1.000 / 1.095 | 1.000 / 1.839 | **0.953 / 2.871** | 0.000 / 9.067 |
| paced_sticky 1 | 1.000 / 1.205 | 1.000 / 1.985 | 1.000 / 2.955 | 0.000 / 8.686 |
| paced_sticky 2 | 1.000 / 1.459 | 1.000 / 2.272 | 1.000 / 3.223 | 0.188 / 7.481 |
| paced_sticky 4 | 1.000 / 2.381 | 1.000 / 2.875 | 1.000 / 3.671 | 0.906 / 5.977 |
| paced_sticky 8 | 1.000 / 4.526 | 1.000 / 4.329 | 1.000 / 4.453 | **0.984 / 5.498** |
| paced_sticky 16 | 1.000 / 6.951 | 1.000 / 6.169 | 1.000 / 5.812 | 1.000 / 5.751 |
| paced_sticky 32 | 1.000 / 7.981 | 1.000 / 7.076 | 1.000 / 6.638 | 1.000 / 6.072 |

加粗为协议规则选中的基线。

## 3. 解读与边界

- N=8 选中的 margin=0.5 刚好达到 F ≥ 0.95（0.953，区间下限 0.891）。换到测试集，可行率可能略低于 0.95；margin=1 的 F=1.000，能耗/上界为 2.955，只高 3%。测试时按协议使用验证集选出的 margin=0.5，不根据测试结果再改。
- paced_sticky 每次选功率只看"剩余比特 ÷ 剩余时隙"和上一时隙的干扰，没有任何跨 UE 协调。N 较大时它和最优之间的差距，主要来自子信道上的冲突与同时发送，这正是 MARL 可以改进的地方。
- 这里只有一组执行种子，区间只反映实例之间的差异。

## 4. 复现

```bash
git checkout da7a482
P="random_max sticky_max lazy_pacing"
for m in 0 0.1 0.25 0.5 1 2 4 8 16 32; do P="$P paced_sticky:margin=$m"; done
for n in 4 6 8 12; do
  uv run --locked python -m dsa_period.experiments.evaluate_policies \
    --config configs/scenario-v1-n$n.toml --split validation --policies $P \
    --output outputs/baseline-calibration-validation-n${n}_20260924 --formal
done
```

汇总在本报告所在的提交上运行：

```bash
uv run --locked python scripts/summarize_policy_evaluation.py \
  outputs/baseline-calibration-validation-n{4,6,8,12}_20260924/summary.json --output <新路径>/summary.json
```
