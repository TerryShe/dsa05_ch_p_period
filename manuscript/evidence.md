# 初稿证据索引与解释边界

本文件用于审稿前的内部核对，不属于论文正文。时间截点为 2026-09-27，撰写起点代码提交为 `4e38cbae6b0d6841d824c8e85b6c0efd444d7f78`。实验训练、评估时使用的历史提交与检查点哈希，以各实验报告及归档 JSON 为准，不能用当前提交替代。

## 1. 名称映射与证据层次

| 论文名称 | 项目名称 | 用途 |
|---|---|---|
| MAPPO-F | A，`fixed-w10` | 主方法，固定违约惩罚 10 |
| MAPPO-D | C，`local-eta2` | 逐 UE 对偶对照，步长 2、上限 50 |
| Global-multiplier variant | B，`global-eta20` | N=8 短训的次要失败观察 |
| Paced-sticky | `paced_sticky` | 最强已评估且满足协议可行性判据的本地规则 |
| `L_0` | 归档中的 lower bound | 孤立单 UE 最小能量之和，要求全部包成功 |
| `U_0` | 归档中的 upper bound | 可重放的周期正交零违约调度；N≤K 时等于 `L_0` |

**正式性能比较**只使用 [EXP-004 测试报告](../docs/results/exp004-test_20260926/report.md)和 [summary.json](../docs/results/exp004-test_20260926/summary.json)。训练曲线、干扰诊断与 EXP-002 是验证集证据，不作为新增测试结果。可靠性尾部统计是 EXP-004 的事后描述性分析。

## 2. 主结论：冻结检查点的测试结果

每个 N：200 个独立测试实例 × 2 次随机执行 × 3 个训练种子。规则基线没有训练种子。以下为 selected 检查点，非 final 检查点。

| N | MAPPO-F 的 F [95% CI] | MAPPO-F 的平均 E/U₀ [95% CI] | paced-sticky 平均 E/U₀ | 相对节能 |
|---|---|---|---|---|
| 4 | 0.993 [0.978, 1.000] | 1.007 [1.004, 1.010] | 1.000 | 无节能优势，约高 0.7% |
| 6 | 0.986 [0.971, 0.997] | 1.094 [1.086, 1.103] | 1.725 | 36.6% |
| 8 | 0.994 [0.982, 1.000] | 1.213 [1.205, 1.221] | 2.798 | 56.7% |
| 12 | 0.984 [0.965, 0.996] | 1.623 [1.428, 1.976] | 5.508 | 70.5% |

节能计算为 `1 − mean(E_MAPPO/U₀) / mean(E_paced/U₀)`；不是逐实例节能百分比的平均，也不是所有实例绝对焦耳数求和后的百分比。图 3、表 II 及摘要数字由原始汇总生成，未从已四舍五入的表格反推。

相对 paced-sticky 的能量配对差（MAPPO-F − paced-sticky）95% 区间分别为：N=6 `[-0.659, -0.603]`，N=8 `[-1.632, -1.541]`，N=12 `[-4.179, -3.535]`。三者均不含零。区间来自冻结评估的 10,000 次两层 bootstrap，实例共同重采样，两方法训练种子独立重采样；每实例两次执行先取平均。代码见 [evaluation.py](../src/dsa_period/evaluation.py)。

`F` 是“回合内所有 UE 的经验违约率均不超过 5%”的回合比例。协议以 `F` 的**点估计**是否达到 0.95 判读；它不要求 95% 区间下限达到 0.95。N=8 的 paced-sticky 和 N=12 的 MAPPO-D 都有低于 0.95 的区间下限。因此不能表述为“以 95% 置信度保证至少 95% 场景可行”。

## 3. 其余结论与定位

| 正文内容 | 证据 | 允许的结论 / 限制 |
|---|---|---|
| 图 1，通信场景与周期分组窗口 | `env.py`、`physics.py`，协议 v1 | 示意图，不是样本拓扑或一次策略执行结果；设备位置和相位仅用于解释 |
| 图 2，网络与训练框架 | `features.py`、`networks.py`、`policy.py`、`trainer.py` | 对应 K=4、5 个正功率档、13 维 token/context 和 64 维 GRU；中央 critic 不参与执行 |
| 图 3、表 II，N=4/6/8/12 主结果 | EXP-004 `summary.json → long → N → methods` | 优于本文评估的规则基线；不能扩展为优于现有所有学习算法 |
| A 与 C 无统计上可分辨的能量差异 | EXP-004 报告“结论”第 3 项及 JSON 差值区间 | 四个 N 的 selected A−C 区间都包含零；不等于证明两者等效 |
| 最后检查点的可靠性差异 | EXP-004 §2.1 | A 的 final F 为 0.987–0.992；C 在 N=8/12 为 0.915/0.938。仅针对本配置 |
| N=8 学习曲线，图 4 | [EXP-001 第 2b 阶段汇总](../docs/results/exp001-phase2b_20260925/summary.json) | 未平滑；C 每个检查点使用当次验证选择的初始乘子。不是固定同一乘子贯穿的学习曲线 |
| N=8 长短训差异 | EXP-004 `short_n8` 与 `long/8` | A：2.321→1.213；训练预算与熵退火同时变化，不能只归因于训练时长 |
| 重叠率与干扰损失，图 5 | [短训诊断](../docs/results/exp001-efficiency-diagnosis_20260925/summary.json)、[长训诊断](../docs/results/exp001-phase2b_20260925/diagnosis.json) | 验证集描述性关联，不能代替 GRU、OFF、功控等因果消融 |
| 显式周期历史特征的负结果 | [EXP-002 报告](../docs/results/exp002-periodic-memory_20260925/report.md)、[汇总](../docs/results/exp002-periodic-memory_20260925/summary.json) | 仅限 2M 预算及 lags=10/20/40；不能说周期性或记忆没有价值 |
| 全局乘子 B 的 F=0.592 | EXP-004 §4，EXP-001 第 1 阶段 | 被选中的是未训练的初始检查点；不能作为全局对偶方法普遍无效的证据 |
| 128/9600 个执行不可行 | EXP-004 §3 | 合并 A、C、四个 N 的 selected 结果；108 个最差 UE 在 75–100 m，83 个周期为 40。事后描述性统计 |

### 对行为诊断的精确定义

- **Co-channel overlap / `collision_share`**：与至少另一个正在发送的 UE 共享子信道的传输次数占比。TIN 模型中，这不等同于丢包概率。
- **Interference loss**：`1 − 总实际互信息 / 相同发送动作下的总无干扰互信息`，不是 packet drop rate。
- **Duty / `duty`**：有待发送分组时选择发送的条件比例。不能将其乘 N 来估计任意 slot 的平均同时发送 UE 数。
- 图 5 的柱是训练种子均值，点是各种子；paced-sticky 与中央参照各只有一份汇总，未把它们伪装为多训练种子。

## 4. 中央参照不能混用的约束

[中央参照报告](../docs/results/protocol-v1-references_20260924/report.md)、[测试参照数据](../docs/results/protocol-v1-references_20260924/test.json)、[实现](../src/dsa_period/central.py)共同支持：在静态、频率平坦信道及指定离散功率菜单下，零违约最优值满足 `L₀ ≤ E₀* ≤ U₀`；N≤K 时三者相等。

学习目标允许逐 UE 5% 的分组违约。因此：

- N=4 的“高于零违约精确参照 0.7%”**不等于**“距离允许 5% 违约问题的全局最优只有 0.7%”。
- 不能直接把 `E/L₀ − 1` 和 `E/U₀ − 1` 写成允许丢包问题的严格最优性区间。
- N=12 的 `E/U₀=1.623` 表示能量高于已实现的零违约正交调度参照 62.3%；该参照本身也未被证明是高负载 TIN 问题的最优解。

正文明确使用 **zero-loss reference**，有意收紧了部分内部报告中不带限定的“精确最优”和“距最优差距”表述。

## 5. 模型、算法与指标的代码对应

| 内容 | 代码 / 配置 |
|---|---|
| SINR、Shannon 服务、功率菜单、路径损耗 | [physics.py](../src/dsa_period/physics.py)、[场景 N=8](../configs/scenario-v1-n8.toml) |
| 分组生命周期、部分观测、边界事件顺序 | [env.py](../src/dsa_period/env.py) |
| 由 RSSI 与私有互信息推断增益、特征、局部乘子 | [features.py](../src/dsa_period/marl/features.py) |
| 子信道共享 GRU、Deep Sets 平均聚合、中央 critic | [networks.py](../src/dsa_period/marl/networks.py) |
| 推断时每 slot 更新隐藏状态、独立 UE 随机流 | [policy.py](../src/dsa_period/marl/policy.py) |
| 团队奖励、训练重置、验证候选、检查点选择 | [trainer.py](../src/dsa_period/marl/trainer.py) |
| PPO clipping、GAE、有效 actor 样本 | [ppo.py](../src/dsa_period/marl/ppo.py) |
| 长训超参数 | [exp001-constrained-mappo-long.toml](../configs/exp001-constrained-mappo-long.toml) |
| 评估实例、随机种子、统计口径 | [协议 v1](../docs/research/marl-protocol-v1.md)、[evaluation.py](../src/dsa_period/evaluation.py) |

正文能量 `E` 汇总截止期位于 `(200,2000]` 的分组的完整寿命发射能量，包括失败分组，也包括这些分组在 warm-up 结束前已消耗的能量。mW/UE 是另一项指标，使用 `[200,2000)` 内的全部 slot 发射能量。两者不能不加区别地相互换算。

执行观察是受限的，但在本理想模型中，精确 RSSI 与私有互信息足以在一次发送后恢复自己的静态增益。因此没有宣称 UE 永远不知道自己的信道；真正缺失的信息包括其他 UE 的私有状态及未来动作。空队列 UE 继续接收观测并更新 GRU。反馈、射频电路和计算能耗没有计入。

## 6. 文献查阅来源

BibTeX 中的条目均有真实论文来源；相关工作的技术描述以原文及作者/机构发布资料为依据。

| BibTeX key | 查阅来源 | 对应作用 |
|---|---|---|
| `lazy2002` | [Stanford 作者版 PDF](https://www-isl.stanford.edu/people/abbas/group/papers_and_pub/TN0802.pdf) | 截止期内延长发送以节能的经典问题基础 |
| `naparstek2019` | [作者 arXiv](https://arxiv.org/abs/1704.02613) | 基于本地反馈与循环网络的分布式频谱接入 |
| `liang2019` | [作者 arXiv](https://arxiv.org/abs/1905.02910) | V2X 中联合频谱、功率及载荷时限 |
| `guo2023` | [作者 arXiv](https://arxiv.org/abs/2202.06345)、[作者出版列表](https://www.liang-seu.net/publications/) | 周期关键任务可靠性约束下的功控 |
| `zhang2025` | [SpectrumX 论文页](https://www.spectrumx.org/publication/multi-agent-reinforcement-learning-for-multi-cell-spectrum-and-power-allocation/) | 多小区循环 MARL 频谱与功率分配 |
| `bagherinejad2025` | [Aalborg 机构论文页](https://vbn.aau.dk/en/publications/drl-based-distributed-joint-sub-band-allocation-and-power-control/)、[原文 PDF](https://vbn.aau.dk/ws/files/776357204/RL_PCSA_XR_IBS_Final.pdf) | 最接近的周期 XR 子网联合子带与功率学习；明确区分其目标和信息集合 |
| `yu2022` | [作者 arXiv](https://arxiv.org/abs/2103.01955)、[NeurIPS 正式论文](https://papers.nips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) | MAPPO 基础算法 |
| `calvo2021` | [作者 arXiv](https://arxiv.org/abs/2102.11941) | 状态增强与局部乘子思路；引用的是此版本 |
| `caire2001` | [EURECOM 论文页](https://www.eurecom.fr/publication/394) | 累积互信息 / 增量冗余服务抽象的背景 |
| `zaheer2017` | [NeurIPS 正式论文页](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | 集合聚合与置换对称结构 |

## 7. 后续实验如何进入下一版

EXP-003 的反事实沉默基线没有完成结果进入本稿。EXP-005 的 XR-SBPC 风格 SAC 比较也没有被预写成正面或负面结果。下一版若加入，必须依据它自身完成的冻结评估、可靠性判据和区间判断；不能仅凭少量训练中间点写“优于文献基线”。

当前最明确的证据缺口是**独立实现的文献学习基线比较**。其次是能区分“历史记忆、主动沉默、功率自适应”贡献的受控消融。图 5 足以说明观察到的行为变化，但不填补后一个缺口。
