# ICC 2027 英文论文初稿

**Learning Energy-Efficient Uplink Access with Asynchronous Packet Deadlines**

中文工作题目：异步分组截止期约束下的节能上行接入学习

- [完整英文 PDF](main.pdf)：**6 页**，标准 IEEE conference、10 pt、US Letter、双栏，含 4 幅图、3 张表和 10 篇参考文献。
- [LaTeX 入口](main.tex)、[参考文献](references.bib)。
- [ICC 2027 官方投稿要求](submission_requirements.md)：初投最多 6 页；官网当前截止日为 **2026-10-02**。
- [证据索引](evidence.md)、[13 份归档的证据快照与来源哈希](data/evidence.json)。
- [测试集结果 CSV](data/test_results.csv)、[确认集结果 CSV](data/confirm1_results.csv)、[确认集配对比较 CSV](data/confirm1_comparisons.csv)。

更新日期：2026-09-27。本版整合已完成的 EXP-001、EXP-002、EXP-004 和 EXP-005。跨负载主结果来自 EXP-004 测试集；与 XR-SBPC 式学习基线的比较来自 EXP-005 独立确认集 C1。全部新增数字、表格和诊断从已有归档生成。

## 论文主线

各 UE 根据本地分组状态、延迟公共 RSSI 和私有传输反馈，学习选择子信道、功率和沉默。主方法 **MAPPO-F** 对应项目 A / `fixed-w10`；局部乘子对照 **MAPPO-D** 对应 C / `local-eta2`。

1. 明确周期分组、累积服务、因果信息边界和逐 UE 可靠性目标。
2. 构造同一离散功率模型下的中央零违约参照，并证明 N≤K 时精确。
3. 说明共享循环 Actor、团队奖励、MAPPO 和局部乘子对照。
4. 报告四个负载的冻结测试结果：N=6/8/12 相比 paced-sticky 的归一化能耗降低 36.6%/56.7%/70.5%。
5. 在新的 N=8 确认集比较六种 SAC 设置：相同能耗奖励下，MAPPO-F 为 1.217，SAC-V2 为 3.028，节能 59.8%；配对能耗差区间不含零。
6. 用描述性诊断解释功率、沉默和干扰的关系，并报告可靠性尾部及比较范围。

SAC 内部的奖励比较固定 γ=0.99；γ=0.5 是敏感性分析。SAC-V2 对 MAPPO-F 比较的是两套完整训练方案，不能单独归因于 PPO、GRU 或中央 critic。局部对偶对照没有统计上可分辨的节能优势。中央参照要求零违约，与学习目标允许 5% 违约的约束不同。

## 文件与图表

| 文件 | 内容 |
|---|---|
| `sections/introduction.tex` | 动机、相关工作、贡献 |
| `sections/model.tex` | 通信与信息模型、能耗和可靠性目标 |
| `sections/references.tex` | 中央能耗参照与命题证明 |
| `sections/method.tex` | 循环 Actor、MAPPO-F/D |
| `sections/results.tex` | 协议、测试集、C1、行为诊断、局限 |
| `sections/conclusion.tex` | 基于已完成实验的结论 |
| `scripts/build_assets.py` | 从 13 份 JSON 生成统计图、两张结果表、数字宏、三个 CSV 和证据快照 |
| `scripts/build_diagrams.py` | 生成场景与网络结构的矢量示意图 |
| `generated/` | `all_methods.tex`、`confirm1_methods.tex`、`numbers.tex` |

| 正文编号 | 图示 | 文件 |
|---|---|---|
| Fig. 1 | 共用 AP、同信道干扰、反馈和异步分组窗口 | [system_scenario.pdf](figures/system_scenario.pdf) |
| Fig. 2 | 共享循环 Actor 与中央训练 | [mappo_framework.pdf](figures/mappo_framework.pdf) |
| Fig. 3 | 四个负载的测试能耗与可行率 | [test_performance.pdf](figures/test_performance.pdf) |
| Fig. 4 | MAPPO 长短训、SAC-V2、规则与中央参照的干扰诊断 | [coordination_n8.pdf](figures/coordination_n8.pdf) |
| 辅助材料，未放入六页正文 | MAPPO-F/D 与 SAC-V2 的 N=8 验证学习曲线 | [learning_n8.pdf](figures/learning_n8.pdf) |

Fig. 1 是模型示意，不是一个实验实例。Fig. 2 由实际网络实现绘制；Actor 在空闲期间仍更新私有隐藏状态。所有图均有 PNG 预览，两张示意图另有可编辑 SVG。

## 构建

从工程根目录运行：

```bash
make -C manuscript
```

使用项目 `.venv/bin/python`、NumPy、Matplotlib、IEEEtran、BibTeX 和 latexmk。也可指定解释器：

```bash
make -C manuscript PYTHON=/absolute/path/to/python
```

只改正文时，可保留已生成图表：

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

构建仅读取已归档结果，不启动训练、策略执行或新评估。中间文件放在 `build/`，不纳入版本控制；只要保留本目录现有图表，脱离实验工程也可编译 PDF。重建图表需要原工程的归档 JSON。BibTeX 保留核查用 URL，PDF 中通过 IEEEtran 的标准选项省略在线地址；状态增强 RL 条目已更新为核实后的 2024 年 TAC 正式版本。

## 当前状态

六页版已编译，并检查了页面、公式与图表、交叉引用、来源数字和字体嵌入。正文采用标准字号和页边距；学习曲线作为辅助材料保留。

正文直接陈述方法、结果和贡献，解释范围集中在实验部分末尾的一段 `Scope and limitations` 中；详细的统计解释与证据边界保留在内部证据索引。

**作者信息尚待补入**。ICC 2027 官网要求 PDF 与 EDAS 的作者名单、题目一致，因此已移除匿名作者标识。当前 PDF 的作者栏为明确的资料占位，尚不能直接上传投稿。作者名单、单位、邮箱及适用致谢填好后需重新检查页数；未执行 EDAS 上传或平台格式认证。分会建议与官方链接见 [投稿要求](submission_requirements.md)。
