# 稿件证据索引与解释边界

更新时间：2026-09-27。本文件用于核对正文，不是论文正文。当前版本纳入已完成的 EXP-005，不能沿用早期初稿“文献学习基线尚未完成”的状态。

## 1. 名称、集合与来源

| 论文名称 | 工程名称 | 含义 |
|---|---|---|
| MAPPO-F | A / `fixed-w10` | 固定违约惩罚 10，主方法 |
| MAPPO-D | C / `local-eta2` | 逐 UE 对偶，步长 2、上限 50 |
| Paced-sticky | `paced_sticky` | 按 N 校准 margin 的本地节能规则 |
| SAC-V1 | `v1-g099`、`v1-g05` | XR-SBPC 式个体可靠性奖励，含中途塑形 |
| SAC-V1s | `v1s-g099`、`v1s-g05` | 只保留完成奖励与过期惩罚 |
| SAC-V2 | `v2-g099`、`v2-g05` | 同一个 SAC 方案，使用 MAPPO-F 的团队能耗奖励 |
| L₀ | `lower` / `lb` | 孤立单 UE 的零违约最小能量之和 |
| U₀ | `upper` / `ub` | 可重放的周期正交零违约调度能量 |

- **跨负载测试**：[EXP-004 报告](../docs/results/exp004-test_20260926/report.md)和[汇总](../docs/results/exp004-test_20260926/summary.json)，N=4/6/8/12，各 200 个实例、2 次执行、3 个训练种子；实例 20,000–20,199。
- **学习基线正式比较**：[EXP-005 报告](../docs/results/exp005-xr-sbpc_20260927/report.md)和[C1 汇总](../docs/results/exp005-xr-sbpc_20260927/confirm1-summary.json)，仅 N=8，200 个新实例、2 次执行、3 个训练种子；实例 30,000–30,199。
- **训练与行为诊断**：均在 64 个验证实例上进行，不标作测试或确认集结果。
- C1 的[实验设计](../docs/experiment-designs/005-xr-sbpc-baseline_20260926.md)在评估前定稿。原测试集已被查看，因此所有新学习比较统一使用 C1；没有依照比较方向决定是否启用确认集。
- 各实验使用的历史提交、选中检查点和 SHA-256 由实验归档记录。论文构建时的代码提交不能替代实验运行的历史提交。

构建脚本读取 13 份现有 JSON，并在 [data/evidence.json](data/evidence.json) 保存路径、SHA-256 和完整内容快照。不会运行训练或策略评估。

## 2. 测试集主结论

以下全部为 selected 检查点。规则没有训练种子；学习方法使用全部三个种子。

| N | MAPPO-F 的 F [95% CI] | MAPPO-F 平均 E/U₀ [95% CI] | paced-sticky 平均 E/U₀ | 相对节能 |
|---|---|---|---|---|
| 4 | 0.993 [0.978, 1.000] | 1.007 [1.004, 1.010] | 1.000 | 无优势，约高 0.7% |
| 6 | 0.986 [0.971, 0.997] | 1.094 [1.086, 1.103] | 1.725 | 36.6% |
| 8 | 0.994 [0.982, 1.000] | 1.213 [1.205, 1.221] | 2.798 | 56.7% |
| 12 | 0.984 [0.965, 0.996] | 1.623 [1.428, 1.976] | 5.508 | 70.5% |

节能为 `1 − mean(E_MAPPO/U₀) / mean(E_baseline/U₀)`，不是逐实例节能百分比的平均，也不是所有实例绝对焦耳数求和后的比例。摘要、数值宏和表格由未舍入的 JSON 生成。

MAPPO-F 减 paced-sticky 的 E/U₀ 配对差区间：N=6 `[-0.659,-0.603]`、N=8 `[-1.632,-1.541]`、N=12 `[-4.179,-3.535]`，均不含零。MAPPO-F 与 MAPPO-D 的差值区间在四个 N 都包含零，不构成等效性证明。

## 3. C1：奖励与完整方案比较

正文 Table III 包含全部六个 SAC 设置以及 MAPPO-F/D 和 paced-sticky。完整均值和区间（含 final 检查点及所有规则）在 [confirm1_results.csv](data/confirm1_results.csv)。配对差在 [confirm1_comparisons.csv](data/confirm1_comparisons.csv)。

| 结论 | C1 证据 | 解释范围 |
|---|---|---|
| 原测试结论复现 | MAPPO-F 的 E/U₀=1.217 [1.209,1.225]，F=0.9925；paced-sticky 为 2.821 | 节能 56.9%；使用独立新实例 |
| 同一 SAC 方案、γ=0.99 下的奖励差异 | V1−V2=4.073 [2.838,5.634]；V1s−V2=4.459 [2.689,5.737]；三者均满足经验判据 | 改动包括可靠性目标、显式能耗以及个体奖励变为团队奖励，不能只归因于能耗项的单独作用 |
| 相同奖励的完整方案比较 | V2−MAPPO-F=1.811 [1.648,1.999]；MAPPO-F 相对 V2 节能 59.8% | 执行信息与奖励相同，但算法、循环结构、价值函数、回放和探索共同改变；没有单组件因果结论 |
| 该比较的可行率差异 | V2−MAPPO-F 的 F 差为 −0.0142 [−0.0358,0.0075] | 未分辨出可行率差异，不能证明相同 |
| 折扣敏感性 | V2、γ=0.5 的 F=0.621；选中的是初始未训练策略 | 不对它与可行方法作能耗优越性判读 |
| 塑形没有分辨出的能耗效应 | V1−V1s：γ=0.99 区间 [−2.347,1.651]；γ=0.5 区间 [−1.157,0.201] | 不能断言塑形普遍无效 |
| 训练不稳定 | V2、γ=0.99：selected 在 3M/5M/7M，C1 final F=0.636 | 主结果依赖预先规定的验证选模 |
| 规则参照处于门限 | paced-sticky F=0.950，380/400，CI [0.9225,0.9725] | 以点估计通过，不是可靠性置信保证 |

γ 的选择只看种子 101 的验证候选，随后应用于该奖励的全部三个种子：V1/V1s 选 0.5，V2 选 0.99。奖励主分析预先固定 γ=0.99，不能用各自调 γ 后的数值宣称纯奖励效应。

SAC 是在本工程信息模型中实现的 XR-SBPC 式基线，**不是原文结果复现**：去除了邻居动作和即时信道输入；加入了空闲转移的 Q 更新并屏蔽空闲 Actor/温度损失；只考察两个折扣，其他参数迁移自原文。比较未覆盖 SAC 的完整调参空间，也没有测试 N≠8。

## 4. 图表和诊断来源

| 正文位置 | 来源 | 边界 |
|---|---|---|
| Fig. 1 场景图 | `env.py`、`physics.py`、场景规范 | 位置和相位为示意，不是某次策略结果 |
| Fig. 2 网络图 | `features.py`、`networks.py`、`policy.py`、`trainer.py` | 13 维 token/context、64 维 GRU；中央 critic 不参与执行 |
| Fig. 3、Table II | EXP-004 `summary.json → long → N → methods` | 同一测试集合，各方法遵循同一经验可行性判据 |
| Table III | EXP-005 `confirm1-summary.json → methods` | 只使用 C1；六种 SAC 设置全数列出 |
| Fig. 4 短训 | EXP-001 效率诊断 `summary.json → results → A-s{seed}` | 验证集，2M 预算 |
| Fig. 4 长训、SAC、规则与中央调度 | EXP-005 `diagnosis-A.json`、`diagnosis-v2.json` | 验证集，SAC-V2 固定 γ=0.99；柱是种子均值，点是各种子 |
| 1M–4M 首次优于规则；SAC 均未达到 | EXP-005 `validation-analysis.json` | 验证诊断，非测试结果；条件同时要求 F≥0.95 |
| 2M 与 20M 的性能差 | EXP-004 `short_n8` 和 `long/8` | 预算和熵退火均改变，不能分离归因 |
| 显式周期历史的负结果 | EXP-002 `summary.json` 与报告 | 仅 lags=10/20/40 和 2M 预算，不能泛化为记忆没有价值 |
| 128/9600 个执行不可行 | EXP-004 报告 §3 | 合并 selected A/C 与四个 N 的事后描述性统计 |

辅助学习曲线 [learning_n8.pdf](figures/learning_n8.pdf) 未放入六页正文：读取 EXP-001 第 2b 阶段 `summary.json` 和 EXP-005 `validation-analysis.json`；显示三个训练种子及其均值，不平滑。MAPPO-D 每个验证点使用当次选定的初始乘子。

精确定义：

- **Co-channel overlap / `collision_share`**：与另一正在发送的 UE 共用子信道的传输次数占比，不等于丢包概率。
- **Interference loss**：`1 − 总实际互信息 / 相同发送动作下的总无干扰互信息`，不是分组违约率。
- **Duty / `duty`**：有待发送分组时选择发送的条件比例，不是全部设备时隙的占用率。
- **完成时间**：V1/V1s 的 0.16–0.27 是各策略平均完成时间除以周期，不是每个包都在前四分之一周期完成。

行为诊断支持低功率、主动沉默、较小干扰与节能的关联；没有独立的 GRU/OFF/功控消融。EXP-003 没有完成结果纳入正文。

## 5. 模型与统计边界

中央参照要求全部包按期完成：同一离散功率模型下 `L₀ ≤ E₀* ≤ U₀`，N≤K 时三者相等。学习策略允许逐 UE 5% 违约，因此 N=4“约高于参照 0.7%”不是允许丢包问题的最优性保证，E/L₀ 或 E/U₀ 也不是该问题的严格近似比。N>K 的正交调度未被证明是 TIN 问题的全局最优。

`F` 是回合内所有 UE 的经验违约率都不超过 5% 的回合比例。协议以点估计 F≥0.95 判读；不能表述为“以 95% 置信度保证 95% 场景可行”。两层 bootstrap 重抽训练种子及实例；实例内两次执行先平均；方法比较共同重抽实例、独立重抽两方法的训练种子。实现见 [evaluation.py](../src/dsa_period/evaluation.py)。

正文能量 E 统计截止期在 `(200,2000]` 的分组的完整寿命能量，包括失败包及这些包在 warm-up 结束前的能量。mW/UE 使用 `[200,2000)` 内所有发射时隙；两项不能直接相互换算。

精确 RSSI 与私有互信息使 UE 在一次发送后可恢复自己的静态增益，因此“局部信息”不等于 UE 永远不知道自身信道。空闲 UE 持续接收反馈、更新 GRU。只计发射能量，不计反馈、电路或计算成本。经验可行性不构成 URLLC 或逐实例长期约束保证。

关键代码：[`physics.py`](../src/dsa_period/physics.py)、[`env.py`](../src/dsa_period/env.py)、[`features.py`](../src/dsa_period/marl/features.py)、[`networks.py`](../src/dsa_period/marl/networks.py)、[`trainer.py`](../src/dsa_period/marl/trainer.py)、[`sac.py`](../src/dsa_period/marl/sac.py)、[`sac_trainer.py`](../src/dsa_period/marl/sac_trainer.py)、[`xr_sbpc.py`](../src/dsa_period/marl/xr_sbpc.py)。参数来自 [MAPPO 长训配置](../configs/exp001-constrained-mappo-long.toml)与 [SAC 配置](../configs/exp005-xr-sbpc-baseline.toml)。

## 6. 文献来源

| BibTeX key | 原文或作者/机构来源 | 用途 |
|---|---|---|
| `lazy2002` | [Stanford 作者版](https://www-isl.stanford.edu/people/abbas/group/papers_and_pub/TN0802.pdf) | 截止期内延长服务以节能 |
| `naparstek2019` | [作者 arXiv](https://arxiv.org/abs/1704.02613) | 本地反馈、循环网络与频谱接入 |
| `liang2019` | [作者 arXiv](https://arxiv.org/abs/1905.02910) | V2X 联合频谱、功率和载荷时限 |
| `guo2023` | [作者 arXiv](https://arxiv.org/abs/2202.06345)、[作者出版列表](https://www.liang-seu.net/publications/) | 周期关键任务可靠性功控 |
| `zhang2025` | [作者论文](https://arxiv.org/abs/2312.05746)、[IEEE ComSoc 2025-08 目录](https://www.comsoc.org/system/files/2025-09/publications_contents_digest_2025_aug.pdf) | 多小区循环 MARL；补全正式页码 5980–5992 |
| `bagherinejad2025` | [Aalborg 机构页](https://vbn.aau.dk/en/publications/drl-based-distributed-joint-sub-band-allocation-and-power-control/)、[作者接收稿](https://vbn.aau.dk/ws/files/776357204/RL_PCSA_XR_IBS_Final.pdf) | XR-SBPC 问题、奖励和方案；本工程差别见上文 |
| `yu2022` | [NeurIPS 原文](https://papers.nips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)、[Berkeley 机构页](https://its.berkeley.edu/node/4211) | MAPPO；补全页码 24611–24624 |
| `calvo2024` | [作者 arXiv](https://arxiv.org/abs/2102.11941)、[正式出版元数据](https://researchportal.ip-paris.fr/fr/publications/state-augmented-constrained-reinforcement-learning-overcoming-the/) | 状态增强与局部乘子；引用改为 TAC 69(7):4275–4290, 2024 |
| `caire2001` | [EURECOM 机构页](https://www.eurecom.fr/publication/394)、[DOI 元数据](https://api.crossref.org/works/10.1109/18.930931) | 累积互信息模型；页码 1971–1988 |
| `zaheer2017` | [NeurIPS 正式论文页](https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) | 集合聚合与置换对称结构 |

本版修正 BibTeX 中 El Gamal 的姓氏解析，补全已核实的正式出版信息。PDF 中不打印冗长的在线地址；URL 和 DOI 仍保留在 `.bib` 及此索引中。
