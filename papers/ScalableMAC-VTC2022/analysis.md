# ScalableMAC-VTC2022：BS 与 UE 联合学习接入及消息内容

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Scalable Joint Learning of Wireless Multiple-Access Policies and their Signaling |
| 作者 | Mateus P. Mota, Alvaro Valcarce, Jean-Marie Gorce |
| 正式出处 | IEEE VTC 2022 Spring Workshops，1–5；[DOI](https://doi.org/10.1109/VTC2022-Spring54318.2022.9860804) |
| 核实来源 | [arXiv 接收说明](https://arxiv.org/abs/2206.03844)；[作者 HAL 清单](https://cv.hal.science/jean-marie-gorce)给出会议与 DOI；单篇 HAL 页面本次被访问保护阻拦 |
| 本地 PDF | [ScalableMAC-VTC2022.pdf](ScalableMAC-VTC2022.pdf)，5 页；arXiv:2206.03844v1，2022-06-08 |
| 下载来源 | [固定版本 PDF](https://arxiv.org/pdf/2206.03844v1)，2026-10-10 下载成功；格式、页数和文本已核验 |
| 阅读范围 | 全文 §I–V、表 I–II 和结果文字；图只作趋势核对，未数字化曲线，未复现代码或逐项核对出版社版 |
| 授权说明 | arXiv non-exclusive-distrib；作者公开稿供本地研究，不等于任意再分发许可 |

## 问题、消息及方法

单小区单共享上行信道，碰撞与固定 TBLER；UE 观测缓冲长度及历史、选择空闲/发送/删包。BS 观测信道空闲、成功发送者、不可解码能量，并与各 UE 交换消息。**BS 和 UE 都是学习智能体**，消息含义不预先约定。实验上行词表 2 个符号、下行 3 个符号，每个 UE 每时隙可各收发一条消息；控制链路专用无误。

MADDPG 的 UE Actor 共享参数且输入不含 UE 身份，集中 Critic 只用于训练，Gumbel-Softmax 处理离散动作。研究 goodput 和碰撞率，不研究按期违约尾部。到达被称作 Poisson，但实现描述为每时隙按概率到达一个包；移植时应按实际离散生成机制核对。

规模实验只到 5 个 UE：2–4 个 UE 优于基线，5 个 UE 的优势下降。结果包含从多次训练中选历史最好协议；不能将此口径与 DSA05 所有种子均值直接对照。

## 与 DSA05 V2 rev3 的重合和区别

**同比特学习消息的直接方法来源，优先级 P0。** 小词表、共享 Actor、无 UE ID、通过 BS 交换消息并协调，已有先例。其 1 比特上行内容由学习决定，足以否定“增加一个可学习比特本身就是创新”。

DSA05 拟采用随活动 UE 附带的紧迫标志与按子信道公共 OR 摘要；Mota 是逐 UE 上下行消息，不是 K 维公共摘要。DSA05 关注最差 UE、时间变化的衰落、多个子信道、离散功率、剩余业务的跨时隙累积与不可靠消息；上述是待实证的区别。

## 应如何构造公平对照（本课题推演）

1. 第一对照保留 DSA05 MAPPO，只增加可学习二元消息头，并施加与紧迫标志完全相同的发送时机、聚合规则、接收门限、时延和缺失掩码。
2. 此对照只检验**固定公共聚合架构中的消息内容**。因其 AP 不学习逐 UE 的下行消息，它不能代表 Mota 原始自由协议的全部表达能力；若要声称胜过一般学习通信，需额外增加可学习 AP 聚合器/广播编码器且保持预算相同。
3. 同时记账总上行有效载荷与下行广播，不将“每次发送 1 比特”误写为全网络每时隙仅 1 比特。若活跃 UE 数为 A_t，理想有效载荷至少是 A_t + K 比特/时隙，再另计编码和资源占用。
4. 报告所有训练种子以及确认集结果；禁止从最优种子结果推断平均稳定性。优先看内容改变是否保护最弱 UE，而非仅总吞吐。

## 去重与引用

其前篇 [The Emergence of Wireless MAC Protocols with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2108.07144) 为 GLOBECOM Workshops 2021；arXiv 明确提示两文有大量文本重合。本次只归档 2022 扩展稿。若需要最早出处，可同时引用前篇，但不要计作两项独立新算法。
