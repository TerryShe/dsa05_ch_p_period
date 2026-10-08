---
doc_id: REG-DSA05-EXP010-SETS
type: protocol-appendix
date: 2026-10-08
protocol: marl-protocol-v2（冻结核心不变）
design: EXP-DSA05-010（2026-10-08 定稿）
manifest: exp010-sets.json
---

# EXP-DSA05-010 的评估切片登记

本附录登记 EXP-DSA05-010（判别实验）使用的开发评估切片，依据[设计](../experiment-designs/010-edf-imitation-diagnostic_20261008.md) §3.1。

- 协议 v2 的冻结核心、EXP-008 与 EXP-009 的登记都不变（重新生成后两份清单逐字节相同）。
- **机器清单**：[exp010-sets.json](exp010-sets.json)，由 `scripts/register_exp008_sets.py` 生成，`tests/test_exp010.py` 核对。

## exp010-eval512

| 项目 | 内容 |
|---|---|
| 实例 | 开发池 4,000–4,511（半开区间 [4000, 4512)），512 个部署 |
| N | 6、8、10 |
| 执行 | 每个部署 1 次，完整 20,200 时隙回合 |
| 种子 | 开发池编码 0，v2 的执行、衰落种子基准与逐 UE 派生，所有方法共用 |
| 用途 | 只用于本实验的判别评估（§3.2 的 D₈ 与描述性比较），不用于任何检查点选择；学生的检查点在 exp009-screen256 上选择 |
| 读取方式 | `evaluate_capacity --split dev --screen exp010-eval512 --grid N --formal`，需干净的已提交工作树 |

- `splits_v2.check_dev_range` 拒绝任何与 4,000–4,511 重叠的普通开发评估；`analyze_exp010.py` 只接受带该切片标记的正式输出。
- 本实验不读取任何确认集，结论是开发阶段的决策依据。

## 登记时的使用核查（2026-10-08）

v2 开发实例只用过 0–63、1,000–1,399、2,000–2,255、3,000–3,255；4,000–4,511 未被使用。第 0 阶段的检查只用训练池与已用过的开发实例 1,000–1,003。
