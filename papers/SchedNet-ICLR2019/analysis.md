# SchedNet-ICLR2019：有限共享媒介下选择谁发送信息

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Learning to Schedule Communication in Multi-agent Reinforcement Learning |
| 作者 | Daewoo Kim, Sangwoo Moon, David Hostallero, Wan Ju Kang, Taeyoung Lee, Kyunghwan Son, Yung Yi |
| 发表 | ICLR 2019；本地 arXiv:1902.01554v1（2019-02-05），首页注明 ICLR 2019 |
| 来源 | [arXiv](https://arxiv.org/abs/1902.01554)；[公开 PDF](https://arxiv.org/pdf/1902.01554)；[会议全文入口](https://openreview.net/pdf?id=HUAnBToP_a) |
| 本地 | [SchedNet-ICLR2019.pdf](SchedNet-ICLR2019.pdf)，17 页（含附录）；会议 PDF 下载返回 403，改用 arXiv，未逐字比对两版 |
| 阅读 | 2026-10-10；§1、§3 全部、§4 结果解释、附录调度实现讨论；未核对全部训练代码与推导 |

## 原文机制

在 CTDE 框架内同时学习消息编码、通信调度权重和环境动作。每步只允许 k 个智能体发送有限大小的消息，采用 Top(k) 或 Softmax(k) 调度；接收者根据收到的消息选动作。实验为协作导航与捕食任务，附录讨论用 CSMA 实现权重调度。

## 对 V2 的意义

“信息预算有限时传最有价值的本地状态”并非新主张。DSA05 的 L 消息头只能作为一种同预算的学习对照，不能代表所有通信 MARL。SchedNet 表明“何时/谁发送”与“发什么”是耦合的：V2 限制标志只能随数据发送，因此还没接入的紧急 UE 不能单独求助，这可能是新的性能瓶颈。

其调度与消息模型不能直接当成免费可用模块：DSA05 的 OR 汇总会丢掉身份和消息数量，而且有一个时隙延迟、SINR 漏读；Top(k) 的权重仲裁也需要资源。原文的抽象预算不能证明 DSA05 的 −10 dB 检测能力。

## 建议

- 将“事件触发紧迫标志”与“每次数据发送都附带消息”作为后续开销—可靠性前沿的可选方向，计入额外控制资源。
- 若做 SchedNet 式比较，应统一最终接收摘要、带宽、仲裁成本及误检模型；直接给它全体消息、再同 U 比较会混淆信息和算法。
- 先完成 EXP-011/012，不建议当前再开一个一般 MARL 通信算法项目。
