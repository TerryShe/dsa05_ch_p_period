# SMART-TCCN2021：紧迫性、自适应接入和公共传输价格

## 基本信息与版本

| 项目 | 内容 |
|---|---|
| 标题 | SMART: Situationally-Aware Multi-Agent Reinforcement Learning-Based Transmissions |
| 作者 | Zhiyuan Jiang, Yan Liu, Jernej Hribar, Luiz A. DaSilva, Sheng Zhou, Zhisheng Niu |
| 发表 | IEEE TCCN，7(4)，1430–1443，2021；DOI [10.1109/TCCN.2021.3068740](https://doi.org/10.1109/TCCN.2021.3068740) |
| 关系 | 原文明确说明部分结果见 SPAWC 2019；期刊版从 DQN 改为 DDPG，并扩展状态/QoS 情形，不能混为同一篇 |
| 来源 | [作者所在实验室全文](https://network.ee.tsinghua.edu.cn/niulab/wp-content/uploads/2022/08/SMART_Situationally-Aware_Multi-Agent_Reinforcement_Learning-Based_Transmissions.pdf)；[Virginia Tech 机构库](https://vtechworks.lib.vt.edu/items/0896ec1d-3c03-4363-814c-a09452c9b6d7) |
| 本地 | `SMART-TCCN2021.pdf`，14 页，出版社排版稿；页脚限制再分发，只保留本地阅读副本并列入 `papers/.gitignore` |
| 阅读 | 2026-10-10；§I–III，§IV 的方法/指标/主要图表，结论；未逐项复算公式和曲线 |

## 原文机制

单信道碰撞接入，时隙中有竞争 mini-slot 和数据阶段。终端按本地状态调节 p-persistent CSMA 的接入概率；状态/代价覆盖 AoI、AoS、交付间隔等 QoS。SMART 离线针对不同传输税训练策略，执行时中心根据接入反馈调整公共传输税，低频广播；终端据本地状态和税选择接入强度。它不是完全无公共控制信令，也不是逐 UE 发布紧迫度。

期刊版不仅给经验曲线，也分析了复杂度、存储、收敛时间和执行成本；这是 DSA05 对目标期刊论文证据厚度的实用参照，不能据此推断录用标准。

## 与 V2 的重合和差异

**“自身紧迫性 + 公共反馈 + 分布式学习减少碰撞”已经覆盖。** 当前框架只提 SPAWC 2019 会低估这条已有研究线，相关工作应优先引用 TCCN 2021 扩展版。

公共税表示整体拥塞/竞争压力；DSA05 的 K 比特摘要表示各子信道近期的紧迫占用，信息语义与时间尺度不同。SMART 主要验证信息新鲜度等代价，不等同于异步周期包在时变衰落、多子信道服务累积下的最差 UE 截止期违约率。

## 对照与可研究问题（本次判断）

- 在已有 U/O/L 之外，“全局拥塞价格”和“逐子信道紧迫摘要”是有意义的信息消融方向，可检验收益到底来自全局降载还是定向保护弱 UE；应另行登记，不能看 EXP-011 结果后偷偷加进原假设。
- 复现税控规则时同时计入广播周期、量化位数和反馈延迟。无需照搬原文所有 DDPG 网络，就能构造解释明确的通信机制对照，但必须称为改编。
- 想要高于“MAPPO 加一个标志”的创新强度，需要说明局部摘要为什么比公共价格更有效，以及在强拥塞、多个紧急 UE 同时出现时为何失效。
