---
doc_id: RES-DSA05-CONFIRM1-REFERENCES
type: reference-archive
status: formal
date: 2026-09-27
protocol: marl-protocol-v1
design: EXP-DSA05-005 §3.2
---

# 确认集 C1 的中央参照（N = 8）

按 [EXP-DSA05-005](../../experiment-designs/005-xr-sbpc-baseline_20260926.md) §3.2 与[协议 v1 附录](../../research/marl-protocol-v1.md)，在确认集 `confirm1`（备用池实例 30,000–30,199）上，为 N = 8 逐实例计算最小能耗的下界（逐 UE 单独最优）与上界（正交分时循环调度，经环境重放核验）。方法与[协议 v1 参照](../protocol-v1-references_20260924/report.md)相同。这些参照是 EXP-DSA05-005 在 C1 上计算 E/LB、E/UB 的分母。

计算时**没有运行任何启发式**（`--skip-heuristics`），也还没有任何方法在 C1 上评估过。

## 文件

| 文件 | 内容 |
|---|---|
| [confirm1.json](confirm1.json) | 汇总；`per_instance` 为逐实例的 `lower_j`、`upper_j`（评价窗口内严格按期的能耗，单位 J）和 `exact`。`evaluation.load_references("confirm1", 8)` 读取此文件 |

汇总记录了原始输出 `outputs/confirm1-bounds-n8_20260927/summary.json`（Git 忽略）的 SHA-256。

## 结果

| 集合 | N | 实例数 | 下界 mW/UE | 上界 mW/UE | 上界/下界（均值 / p90 / 最大） | 精确实例 | 子信道占用率 |
|---|---|---|---|---|---|---|---|
| C1 | 8 | 200 | 3.953 | 4.510 | 1.124 / 1.228 / 1.291 | 10 | 0.994 |
| 测试（对照） | 8 | 200 | 3.868 | 4.382 | 1.113 / 1.232 / 1.284 | 9 | 0.994 |
| 验证（对照） | 8 | 64 | 3.816 | 4.344 | 1.121 / 1.212 / 1.264 | 5 | 0.994 |

全部 200 个实例都求到上界，核验全部通过，MILP 最长 4.0 秒。上界/下界的分布与测试集、验证集一致，C1 的实例在这一指标上没有明显不同。

## 来源与复现

- 正式运行：提交 `b90654c`（EXP-DSA05-005 第 0 阶段），工作区干净。

```bash
git checkout b90654c
uv run --locked python -m dsa_period.experiments.central_bounds \
  --config configs/scenario-v1-n8.toml --split confirm1 --skip-heuristics \
  --output outputs/confirm1-bounds-n8_20260927 --formal
# 汇总：在本存档所在的提交上
uv run --locked python scripts/summarize_central_bounds.py \
  outputs/confirm1-bounds-n8_20260927/summary.json --output <新路径>/confirm1.json
```

原始输出中含有 MILP 的求解耗时，所以重跑的文件不会逐字节一致；其余字段应完全一致。
