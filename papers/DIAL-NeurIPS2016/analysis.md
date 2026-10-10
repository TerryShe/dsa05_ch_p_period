# DIAL-NeurIPS2016：可学习离散消息与训练梯度

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Learning to Communicate with Deep Multi-Agent Reinforcement Learning |
| 作者 | Jakob N. Foerster, Yannis M. Assael, Nando de Freitas, Shimon Whiteson |
| 发表 | Advances in Neural Information Processing Systems 29（NIPS 2016） |
| 来源 | [会议页面](https://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning)；[会议 PDF](https://proceedings.neurips.cc/paper_files/paper/2016/file/c7635bfd99248a2cdef8249ef7bfbef4-Paper.pdf) |
| 本地 | [DIAL-NeurIPS2016.pdf](DIAL-NeurIPS2016.pdf)，会议主文 9 页；不是另附补充材料 |
| 阅读 | 2026-10-10；§4–5 设定/算法、§6 的任务与结果解释；未复现，未逐项核对补充算法 |

## 原文机制

RIAL 将环境动作和通信动作通过循环 Q 学习训练；DIAL 在集中训练时允许经消息通道反向传播，采用带噪连续消息，执行时离散化。任务包含 1 比特通信；实验表明训练方式与正则化会影响能否学出有效协议。这里的训练噪声是学习机制，不是物理 SINR 接收模型。

## 对 DSA05 的约束

EXP-011 的 L 使用 PPO 消息头，不能称为 DIAL 复现。若 L 输给手工紧迫标志，只能说明该训练器/预算未找到更好的消息，不能证明紧迫标志是最优语义，或者学习通信没有价值。此文为这一不对称解释提供直接方法背景。

若以后试 DIAL 式训练，必须处理真实执行链路中的 OR 汇总、OFF 时不能发送、一个时隙延迟和检测误差；连续训练通道不能在评估时偷带额外精度，也不能让发送者得到它原本没有的全局状态。

优先级为方法背景，低于 DCGFRA、XR-SBPC、ECPO 和实际短控制信令。它支持设计更强的学习消息对照，不构成当前 V2 的直接无线基线或可靠性保证。
