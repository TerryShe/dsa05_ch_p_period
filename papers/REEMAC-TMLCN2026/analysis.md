# REEMAC-TMLCN2026：工业上行可靠节能 MAC 与资源级反馈

## 元数据与获取状态

| 项目 | 内容 |
|---|---|
| 标题 | Reliable and Energy-Efficient MAC Protocols in Industrial IoT Networks via Multi-Agent Reinforcement Learning |
| 作者 | Luciano Miuccio, Daniela Panno, Salvatore Riolo, Anita Schilirò |
| 发表 | IEEE Transactions on Machine Learning in Communications and Networking，4，677–705，2026；[DOI 10.1109/TMLCN.2026.3683651](https://doi.org/10.1109/TMLCN.2026.3683651) |
| 原始记录 | [IEEE 文号11481086](https://ieeexplore.ieee.org/document/11481086/)；[作者机构记录](https://www.iris.unict.it/handle/20.500.11769/718789)；[出版社登记元数据](https://api.crossref.org/works/10.1109/TMLCN.2026.3683651) |
| 阅读来源 | [作者在 ResearchGate 公开的原文网页](https://www.researchgate.net/publication/404142417_Reliable_and_Energy-Efficient_MAC_Protocols_in_Industrial_IoT_Networks_via_Multi-Agent_Reinforcement_Learning)，29页正文文本；不是引用网站自动生成的综述 |
| 阅读范围 | 2026-10-10；标题页、§II-A、§III、§IV、§V-D/E 的模型/奖励/控制丢失/消融；未逐图核验，未复现 |
| **本地 PDF** | **尚未取得，不能计入已下载数量**。机构无附件；IEEE正文/PDF返回验证页或HTTP418；作者公开全文两个正常下载链接均HTTP403。没有保存HTML冒充PDF，也未绕过访问限制 |
| 许可 | Crossref出版社登记为CC BY 4.0；可获取性与许可是两个问题 |

## 已核实的内容

工业上行多个频域资源、设备本地策略；FD-MAPPO 给设备独立 actor/critic，BS 的逐资源反馈和本地动作构成本地奖励。执行阶段仍涉及协议控制消息。§II-A 在初始时刻装入有限包，同资源碰撞即失败；目标是可靠清空数据并缩短活跃时间。它不是 DSA05 的异步周期持续到达、TIN 服务累积及最差 UE 截止期违约模型。

§V-D 已考察报头、奖励广播、下行控制丢失，§V-E 分析奖励消融。因此“资源级 AP 广播 + 分布式 MARL”“研究信令丢失”都不能单独成为 DSA05 的新颖性主张。

## 对 V2 的判断与建议

**P0，写论文前必须精读的近期近邻。** 当前可能保留的区别是低 SNR 下紧迫内容的压缩和检测、同预算内容对照、跨时隙截止期与最差 UE 保护，且必须有真实资源代价。它的广播主要服务于分布式训练奖励，不能未经核对便说与 U 的执行摘要完全相同。

建议取得 PDF 后优先核对 pp.681–689、699–703：反馈是否进入执行观测、损失模型是否内容/信道相关、控制开销是否扣减数据资源、丢包是独立还是突发。不要把该文的丢失百分比直接作为 DSA05 SINR 门限，也不要跨指标比较可靠性数值。

获取后应存为 `REEMAC-TMLCN2026.pdf`，重新核对版本和图表，并更新本页下载状态。当前分析明确限于已读原文网页，不代表 PDF 已归档。
