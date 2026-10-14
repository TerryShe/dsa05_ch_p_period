# HRL-URLLC-PIMRC2023：以服务可用度（生存时间）为目标的分层 RL 编排

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Communication-Efficient Orchestrations for URLLC Service via Hierarchical Reinforcement Learning |
| 作者 | Wei Shi, Milad Ganjalizadeh, Hossein Shokri Ghadikolaei（Ericsson Research）, Marina Petrova（KTH、RWTH Aachen） |
| 来源 | IEEE PIMRC 2023（34th，arXiv 页面注明已接收）；arXiv:2307.13415（v1，2023-07-25） |
| 本地 PDF | [HRL-URLLC-PIMRC2023.pdf](HRL-URLLC-PIMRC2023.pdf)，6 页。PDF 页脚为 IEEE 版权声明（仅限个人使用），**只保留本地副本，列入 `papers/.gitignore`**；下载地址 https://arxiv.org/pdf/2307.13415 |
| 阅读范围 | 2026-10-14；摘要、§I、§II（系统模型与服务指标的定义）；分层框架与仿真只看了概要 |

## 研究问题

- 工厂自动化：多个 gNB，各自服务若干工业设备；3GPP InF-DH 信道。
- **服务指标按生存时间定义**：引用 TS 22.104，生存时间 T_s 是应用在收不到预期包时仍能工作的时长；连续 T_s 内没有包到达即为通信服务失败。由此定义每个设备的通信服务可用度与可靠性。
- 决策：每个设备的下行发射功率与最大 HARQ 重传次数（"可靠性增强特征"的编排）。

## 方法

- 多智能体分层 RL：控制回路较快的智能体部署在 gNB，较慢的在边缘或核心网，提供高层指引。
- 目的是减少训练与推断时的信令和时延；性能与单智能体 RL 相当。
- 所解决的问题沿用同组的前作：Ganjalizadeh 等，"Saving Energy and Spectrum in Enabling URLLC Services: A Scalable RL Solution"，IEEE TII 19(10):10265–10276, 2023（DRL 按设备优化 HARQ 重传次数与发射功率，以可用度与可靠性为目标）。**前作 PDF 未取得**（未找到 arXiv 版本）。

## 与 DSA05 的关系（生存时间方向）

**相同**
- **以生存时间定义的服务可用度作为 RL 的目标，已有工作**（Ericsson/KTH，期刊版在 TII 2023）。论文不能宣称"首次把生存时间引入 RL 资源管理"。
- 工厂自动化场景，周期性的控制业务。

**不同**
- **网络侧决策**：gNB 与上层的智能体为各设备配置功率与重传次数；不是设备自主的分布式接入。
- 下行，资源由 gNB 调度，设备之间没有自主竞争。
- 决策变量是可靠性增强特征的配置（功率、HARQ 次数），不是逐时隙的子信道与发送时机。

## 引用建议

- 生存时间方向的相关工作中必须引用（连同 TII 2023 前作），说明"生存时间/可用度 + RL"在网络侧已有研究；我们的区别是上行、设备自主、共享资源的分布式协调。
