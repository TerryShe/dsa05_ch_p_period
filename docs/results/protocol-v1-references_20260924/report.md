---
doc_id: RES-DSA05-PROTOCOL-V1-REFERENCES
type: reference-archive
status: formal
date: 2026-09-24
protocol: marl-protocol-v1
---

# 协议 v1 验证集与测试集的中央参照

按 [MARL 实验协议 v1](../../research/marl-protocol-v1.md) §6，在验证集（编号 10,000–10,063）和测试集（20,000–20,199）上，为四个 N 逐实例计算最小能耗的下界（逐 UE 单独最优）与上界（正交分时循环调度，经环境重放核验）。方法与 [N=4/6/8/12 上下界报告](../central-bounds_20260924/report.md) 相同。这些参照是以后 MARL 与基线计算 E/LB、E/UB 时的分母。

计算时**没有运行任何启发式**（`--skip-heuristics`），所以存档过程没有查看任何方法在测试集上的表现。

## 文件

| 文件 | 内容 |
|---|---|
| [validation.json](validation.json) | 四个 N 的验证集汇总；`per_instance` 为逐实例的 `lower_j`、`upper_j`（评价窗口内严格按期的能耗，单位 J）和 `exact` |
| [test.json](test.json) | 同上，测试集 |

每个汇总记录了原始输出 `outputs/protocol-v1-bounds-{split}-n{N}_20260924/summary.json`（Git 忽略）的 SHA-256。

## 结果

| 集合 | N | 实例数 | 下界 mW/UE | 上界 mW/UE | 上界/下界（均值 / p90 / 最大） | 精确实例 | 子信道占用率 |
|---|---|---|---|---|---|---|---|
| 验证 | 4 | 64 | 3.706 | 3.706 | 1.000 / 1.000 / 1.000 | 64 | 0.689 |
| 验证 | 6 | 64 | 3.734 | 3.833 | 1.021 / 1.070 / 1.096 | 30 | 0.940 |
| 验证 | 8 | 64 | 3.816 | 4.344 | 1.121 / 1.212 / 1.264 | 5 | 0.994 |
| 验证 | 12 | 64 | 3.915 | 5.688 | 1.439 / 1.568 / 1.594 | 0 | 1.000 |
| 测试 | 4 | 200 | 3.823 | 3.823 | 1.000 / 1.000 / 1.000 | 200 | 0.679 |
| 测试 | 6 | 200 | 3.828 | 3.944 | 1.022 / 1.073 / 1.145 | 94 | 0.917 |
| 测试 | 8 | 200 | 3.868 | 4.382 | 1.113 / 1.232 / 1.284 | 9 | 0.994 |
| 测试 | 12 | 200 | 3.838 | 5.494 | 1.413 / 1.554 / 1.624 | 0 | 1.000 |

全部 1,056 个实例的核验都通过，所有 MILP 均求到最优，每个重放包的实测能耗都等于计划值。数值与开发池 0–99 上的结果（上下界报告）一致。

## 来源与复现

- 正式运行：提交 `dbc0889`（协议 v1 冻结提交），工作区干净。
- 验证集、测试集实例的哈希与[协议清单](../../research/marl-protocol-v1.json)一致（`tests/test_splits.py`）。

```bash
git checkout dbc0889
for n in 4 6 8 12; do for split in validation test; do
  uv run --locked python -m dsa_period.experiments.central_bounds \
    --config configs/scenario-v1-n$n.toml --split $split --skip-heuristics \
    --output outputs/protocol-v1-bounds-$split-n${n}_20260924 --formal
done; done
```

汇总在本存档所在的提交上运行：

```bash
uv run --locked python scripts/summarize_central_bounds.py \
  outputs/protocol-v1-bounds-test-n{4,6,8,12}_20260924/summary.json --output <新路径>/test.json
```

原始输出中含有 MILP 的求解耗时，因此重跑的文件不会逐字节一致；其余字段与上下界报告中的复现核验相同，应完全一致。
