---
doc_id: REG-DSA05-EXP008-SETS
type: protocol-appendix
date: 2026-10-03
protocol: marl-protocol-v2（冻结核心不变）
design: EXP-DSA05-008（2026-10-03 定稿）
manifest: exp008-sets.json
---

# EXP-DSA05-008 的筛选集与确认集登记

本附录登记 EXP-DSA05-008 使用的两个固定实例集合，依据[设计](../experiment-designs/008-training-signal_20261002.md) §4.1 与 §7.1。

- **不改动协议 v2 的冻结核心**：划分区间、集合编码、`EVALUATION_REPEATS`、种子基准、`marl-protocol-v2.json` 及其守护测试都保持原样。
- **机器清单**：[exp008-sets.json](exp008-sets.json)，由 `scripts/register_exp008_sets.py` 生成。`tests/test_exp008.py` 会重新生成清单，并逐字节核对其中的范围、重复数、编码与全部摘要。
- **摘要的内容**：每个 N 的摘要覆盖拓扑、阴影、业务数组和第 0–7 时隙的衰落（`splits_v2.instance_digest`）。生成摘要时不运行任何策略。

## 1. 筛选集 exp008-screen256

| 项目 | 内容 |
|---|---|
| 实例 | 开发池 2,000–2,255（半开区间 [2000, 2256)），256 个部署 |
| N | 6、8 |
| 执行 | 每个部署 1 次，完整 20,200 时隙回合，评估窗口与协议 v2 相同 |
| 种子 | 开发池编码 0，v2 的执行、衰落种子基准与逐 UE 派生，所有方法共用 |
| 得分 | S_near = [F_{0.01}(6) + F_{0.01}(8)] / 2 |
| 用途 | 只用于检查点选择与第 1、2 轮筛选（设计 §4），不作为论文结论 |
| 读取方式 | `evaluate_capacity --split dev --screen exp008-screen256 --formal`，需干净的已提交工作树 |

代码中的保护：
- `splits_v2.check_dev_range` 拒绝任何与 2,000–2,255 重叠的普通开发评估（`evaluate_capacity`、`evaluate_capacity_bounds`、`scripts/diagnose_failures_v2.py`）。
- `--screen` 只能读取登记的实例与 N，且必须 `--formal`。
- `select_exp008` 只接受带有该筛选集标记、实例与 N 完全一致的正式输出作为主得分。

筛选实例不进入训练池，不根据其中的难例重新采样训练，也不在看到结果后替换或扩充。

## 2. 确认集 v2-confirm1

| 项目 | 内容 |
|---|---|
| 实例 | 备用池 30,000–30,499（半开区间 [30000, 30500)），500 个部署 |
| N | 协议网格 {2, 4, …, 32}；N_EDF 另加 36、40 |
| 执行 | 每个部署 1 次（`splits_v2.CONFIRMATION_REPEATS = 1`） |
| 种子 | 备用池编码 3，v2 的执行、衰落种子基准；逐 UE 派生、不含 N，因此同一 UE 的种子跨 N 一致 |
| 用途 | EXP-008 的最终比较（设计 §7），冻结后只读一次 |
| 读取方式 | `evaluate_capacity --split v2-confirm1 --confirm-c1 --formal`，需干净的已提交工作树 |

- v1 的确认集 confirm1（备用池 30,000–30,199，v1 场景与种子）不受影响。在 v2 中，`splits_v2` 不接受名为 confirm1 的集合。
- 编号相同不代表实例相同：v2 的拓扑与相位种子基准不同，所以 v1 读过的这些编号不算 v2 已读。
- 读取前须冻结：
  - 代码提交、全部配置；
  - 模型路径与哈希；
  - A-near 的重选结果；
  - 方法清单（`scripts/analyze_exp008_confirm.py` 的 methods 文件）。
- 可以分批计算，但不能在中途查看性能后改变模型、指标或样本数。

## 3. 登记时的使用核查（2026-10-03）

- v2 开发实例只用过 0–63（第 0 阶段量级检查）与 1,000–1,399（第 0 步失败分解）。
- v2 下没有任何运行读取过备用池实例。
- 第 0 阶段的吞吐测量只使用上述已用过的开发实例与训练池。

## 4. 改登记（2026-10-10，用户决定）

- EXP-008 按其 §6 在第 2 轮后结束，没有做长训，**v2-confirm1 从未读取**。
- 2026-10-10 核查：`outputs/` 中没有任何以 v2-confirm1 为集合的评估输出或日志；v2 下读过的备用池实例只有 v2-confirm2（30,500–30,999，EXP-009）。
- 用户决定：v2-confirm1 改作研究框架修订（[FRM-DSA05-RELIABILITY](research-framing_20261010.md)）的**最终确认集**，所有方法只读一次。
- 实例、执行次数、种子与读取开关（`--confirm-c1 --formal`）不变。读取前须按框架 §8 另立协议修订，冻结指标、成败判据、代码提交与方法清单。
- 第 2 节所写的用途（EXP-008 的最终比较）就此作废；A-near 等 EXP-008 专用的冻结项不再适用。

