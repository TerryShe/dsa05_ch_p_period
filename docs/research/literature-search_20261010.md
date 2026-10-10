---
doc_id: LIT-DSA05-SEARCH-20261010
type: search-log
date: 2026-10-10
---

# V2 文献检索范围、筛选与获取记录

综合判断见 [研究评估](dsa05-state-novelty-literature_20261010.md)，逐篇入口见 [papers 索引](../../papers/README.md)，文件校验见 [下载清单](literature-downloads_20261010.json)。本次新增15篇分析，14篇PDF成功，1篇PDF待获取；并更新DCGFRA、XR-SBPC、MARLSPA的V2定位。不是系统综述，不据“本次没有找到”推断某个机制从未被研究。

## 1. 本地优先的检索顺序

先检查 papers/README 的全部43个既有条目和整理规则；重点读 DCGFRA、XR-SBPC、MARLSPA、CG-URLLC、PT-CD、Hou–Kumar、TimelyFlows、LazySched、CalcEE、FRL-EE 等既有分析。再读问题侧及固定OBS文献综述、研究框架rev3和EXP-011设计，辨别V1节能历史定位与当前可靠性目标。

在线优先查作者提交的arXiv、会议官网、作者主页、机构库和出版社。搜索结果中的聚合页面只用于定位；REEMAC的ResearchGate来源是作者上传的论文原文页面，记录为原文网页阅读，未当成已下载PDF。

## 2. 搜索族与筛选原则

| 搜索族 | 代表查询 | 进入本次归档的条目 |
|---|---|---|
| 上行截止期与紧迫接入 | `Robaglia sequential deep Q learning random access deadline`；`Jiang SMART urgency random access` | SeqDQN；SMART期刊扩展 |
| 学习MAC/控制信令 | `Valcarce Hoydis MAC signaling`；Mota/Mostafa/Uyoata准确标题与arXiv号 | MACSignaling；ScalableMAC；ECPO；LPPC |
| 近期资源广播与可靠MAC | `reliable energy-efficient MAC industrial MARL 2026`；`deadline constrained distributed scheduling subcarrier power` | REEMAC（待PDF）；PGDS |
| 有限通信与机制诊断 | `learning to schedule communication SchedNet`；`DIAL communication`；`Pitfalls measuring emergent communication` | SchedNet；DIAL；Pitfalls |
| 控制接收/短包边界 | `URLLC control signaling reliability short packets detection decoding`；`Short-Packet Communications Durisi fading` | SP-Detect；SP-Fading |
| 周期配置授权 | `configured grant time sensitive uplink scheduling periodic traffic 2024 2025` | CG-TSN/C-FAST |
| 衰落截止期理论 | `Zafer Modiano minimum energy transmission fading deadline`；`timely throughput limited feedback wireless` | FadingEE；另列LFDO/PDMax |
| 边界排查 | `deadline urgency multi-channel reinforcement learning wireless 2025 2026`；`urgency one-bit scheduling wireless` | 返回不少专利、综述和非直接模型；没有据这些结果制造“首创”结论 |

选择标准是与当前信息结构、通信机制、接收假设或强基线直接相关，不为凑近年数量而加入不相干大模型/语义通信文献。Mota2021会议稿与2022扩展高度重合，只收2022；SMART优先2021期刊；PGDS按已核实2026正式年份命名，但明确本地是2024预印本。初始arXiv年份、online-first和正式卷期年份分别记录。

## 3. 实际获取与验证

所有成功文件均检查PDF签名、`pdfinfo`页数、可抽取文本和SHA-256，未保存错误页。每篇analysis注明实际读过的章节，不将阅读模型/结论说成复现全部证明/实验。

- SeqDQN、SMART来自作者主页/实验室；DIAL、Pitfalls来自会议官网。
- SchedNet的OpenReview下载返回403，改用公开arXiv v1，未声称两份逐字相同。
- FadingEE的MIT DSpace正式版下载返回405，改用作者J58预印本。
- CG-TSN来自作者机构公开稿，CC BY-NC-ND；PGDS的arXiv稿为CC BY-NC-SA；未修改这些PDF。
- MACSignaling/ScalableMAC为arXiv公开稿，arXiv分发许可不是任意再出版授权。SMART和Pitfalls按库规则仅保留本地副本并忽略Git。
- REEMAC：出版社登记与作者机构核实了书目和CC BY许可，原文网页可读；机构无附件，IEEE正文/PDF返回验证页或418，作者公开PDF链接返回403。**未取得PDF**，没有绕过网站验证或伪造文件。[获取记录和原始入口](../../papers/REEMAC-TMLCN2026/analysis.md)

## 4. 有价值但本次未完成全文归档的方向

以下是有界检索的后续线索，不能当作已读全文或已复现；不是当前EXP-011必须全部复现的清单。

| 优先级 | 论文 / 原始入口 | 当前阅读范围与用途 |
|---|---|---|
| P1 | [Emerging Generalized Wireless MAC Communication Protocols via Abstraction](https://oulurepo.oulu.fi/handle/10024/58238)，OJCOMS 2025，6:6842–6865 | 机构记录与摘要；泛化到UE/包数/信道变化，适合回应跨规模能力；优先于同线GLOBECOM2022前作 |
| P1 | [Is Deadline Oblivious Scheduling Efficient for Controlling Real-Time Traffic in Cellular Downlink Systems?](https://arxiv.org/abs/2002.06474)，2020，DO/LFDO | 原始摘要；及时吞吐、公平与在线调度，可启发比EDF更强的非学习参照 |
| P1 | [Wireless scheduling with deadline and power constraints](https://www.sciencedirect.com/science/article/pii/S0166531620300882)，PDMax | 出版社摘要；deadline/功率/MaxWeight，完整定理及假设尚待核验，不能当DSA05上界 |
| P1 | [5G Control Channel Design for Ultra-Reliable Low-Latency Communications](https://arxiv.org/abs/1803.04139)，2018 | 原始摘要；控制信令与重传工程依据，具体检测分析本次优先读Bana2018 |
| P2 | [Toward Massive, Ultrareliable, and Low-Latency Wireless Communication With Short Packets](https://doi.org/10.1109/JPROC.2016.2537298)，Proceedings of the IEEE 2016 | 作者目录/书目；适合作为短包引言综述，当前已优先下载具体TCOM边界工作 |
| P2 | [GCN-Driven Reinforcement Learning for Probabilistic Real-Time Guarantees in Industrial URLLC](https://arxiv.org/abs/2506.15011)，2025；[IEEE记录](https://ieeexplore.ieee.org/document/11267668/) | 原始摘要与出版记录；多小区干扰图、实时调度，尚未读模型，不能声称具有同样摘要机制 |
| P2 | [Optimizing Wireless Discontinuous Reception via MAC Signaling Learning](https://arxiv.org/abs/2406.13834)，2024 | 原始摘要；DRX/节能和MAC控制时机，对当前最差UE截止期保护较远 |
| P2 | [Grant-Free Radio Access for Short-Packet Communications over 5G Networks](https://arxiv.org/abs/1709.02179)，2017预印本 | 原始摘要；异步接入/重复/SIC，接收机与DSA05的TIN不同 |

黑突发优先接入、CommNet/TarMAC、有限块长的更近期鞍点/非相干模型仍是可追溯方向。本次已用SMART、DIAL/SchedNet和两篇短包论文补主要边界，没有宣称所有这些文献均已检索完毕。

## 5. 本次证据带来的定位修正

PGDS已有广播后价值退让；ECPO已有多信道截止期通信学习；REEMAC已有控制丢失敏感性；Uyoata学习的是预定义消息的发送时机；Pitfalls说明测试时隐去消息的解释有限。新颖性应收紧到**紧迫摘要的具体压缩语义、真实接收与开销，以及最差UE的截止期收益**，并由EXP-011/012和新确认集来证实。
