# DLMA-JSAC2019：深度强化学习多址接入

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Deep-Reinforcement Learning Multiple Access for Heterogeneous Wireless Networks |
| 作者 | Yiding Yu, Taotao Wang, Soung Chang Liew |
| 来源 | IEEE JSAC, vol. 37, no. 6, 2019；arXiv:1712.00162 |
| 本地 PDF | [DLMA-JSAC2019.pdf](DLMA-JSAC2019.pdf)（12 页） |
| 阅读范围 | 摘要、引言、相关工作、第 II 节开头的系统描述 |

## 问题与方法

- 受 DARPA 频谱协作挑战赛（SC2）启发：多个使用不同 MAC 协议的时隙网络共享一个信道，同一时隙多个节点同时发送即碰撞。
- DLMA 节点不知道其他网络用的是 TDMA 还是 ALOHA，只通过自己的动作与观测（成功、碰撞、空闲）学习，用 DQN（ResNet）决定每个时隙是否发送。
- 目标可以是总吞吐或 α 公平。作者为此把 Q 函数与目标函数分开，推广了 Q 学习的框架。
- 结果：与 TDMA、ALOHA 及其混合共存时，DLMA 接近最优的总吞吐与比例公平，而且收敛快、对参数不敏感。

## 与 DSA05 的关系

- **"学习到的接入策略可以利用他人的时间结构"**：DLMA 学会了钻 TDMA 节点的空隙。DSA05 的周期业务也有稳定的时间结构，长训后的策略学会错开发送，与此一致。
- **主要差别**：
  1. 单信道碰撞模型、无功率控制、饱和业务；
  2. 只有一个学习节点（或少量同质节点）与固定协议共存；DSA05 是所有 UE 同时学习的合作问题；
  3. 目标为吞吐或公平，不涉及截止期和能耗。
- [TSRA-2022](../TSRA-2022/analysis.md) 在截止期场景下以 DLMA 为基线，说明这条研究线已经推进到截止期约束。

## 引用建议

- 相关工作：深度强化学习 MAC 协议的代表。
