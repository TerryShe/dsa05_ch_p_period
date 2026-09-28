# ICC 论文完整初稿

**题目：Learning Energy-Efficient Uplink Access with Asynchronous Packet Deadlines**  
中文工作题目：异步分组截止期约束下的节能上行接入学习

- [完整英文 PDF](main.pdf)：7 页，IEEE 双栏会议格式，含 5 幅图、2 张表和 10 篇参考文献。
- [LaTeX 入口](main.tex)、[参考文献](references.bib)。
- [证据与结论索引](evidence.md)：每项结论的数据来源、解释边界和后续补充位置。
- [结果 CSV](data/test_results.csv)、[证据快照](data/evidence.json)：便于核对数字。

初稿日期：2026-09-27。根据已完成的 EXP-001、EXP-002 和 EXP-004 撰写；正式性能结论来自 EXP-004 冻结模型的测试集评估。EXP-005 的完整比较结果尚未纳入。生成本稿仅读取已有归档，不启动训练、策略评估或新的实验。

## 文章主线

在具有异步到达相位的周期业务中，各 UE 仅凭本地分组状态、延迟的公共 RSSI 和私有传输反馈，学习选择子信道、功率和沉默。共享循环 MAPPO 在较高负载下，比经过负载校准的 paced-sticky 本地规则显著降低发射能耗，同时保持较高的“所有 UE 都满足经验违约率限制”的回合比例。

主方法采用固定惩罚，论文名称为 **MAPPO-F**，对应项目的 **A / fixed-w10**。局部乘子对照称 **MAPPO-D**，对应 **C / local-eta2**。当前证据没有显示局部对偶处理带来统计上可分辨的节能优势，因此没有把它作为主要创新。

论证依次是：

1. 定义周期分组、累积互信息和因果观测边界。
2. 构造同一离散功率模型下的中央零违约参照，并给出低负载精确性的证明。
3. 说明共享循环策略、训练奖励及固定惩罚和局部对偶两种处理。
4. 报告冻结检查点在独立测试集的节能、可行率和置信区间。
5. 用验证集诊断解释收益与干扰减少的关联，报告训练成本与残余失败。

本稿把“场景、信息边界、参照体系与实验结论”作为贡献组合，没有声称首次提出联合资源分配、循环 MARL、MAPPO 或对偶更新。

## 内容结构

| 文件 | 内容 |
|---|---|
| `sections/introduction.tex` | 问题动机、相关工作、贡献 |
| `sections/model.tex` | 通信模型、观测、能耗与可靠性目标 |
| `sections/references.tex` | 离散功率中央零违约上下参照与命题证明 |
| `sections/method.tex` | 共享 GRU 策略、MAPPO、奖励与局部乘子 |
| `sections/results.tex` | 协议、规则基线、测试结果、学习与行为诊断、局限 |
| `sections/conclusion.tex` | 结论与后续方向 |
| `scripts/build_assets.py` | 从归档 JSON 生成图、表、数字宏和证据快照 |
| `scripts/build_diagrams.py` | 生成通信场景及网络结构的矢量示意图 |
| `figures/` | 5 幅矢量 PDF 及 PNG 预览；两张示意图另附 SVG |
| `generated/` | 自动生成的结果表与数字宏 |
| `data/` | 来源 SHA-256、归档快照及结果 CSV |

## 图示安排

| 编号 | 内容与用途 | 文件 |
|---|---|---|
| Fig. 1 | 通信场景：共用 AP、同信道干扰、沉默期间的反馈，以及不同周期和相位的分组窗口 | [PDF](figures/system_scenario.pdf) · [SVG](figures/system_scenario.svg) |
| Fig. 2 | 方法网络：局部输入、共享编码器和 GRU、Deep Sets 聚合、功率与 OFF 分支、动作掩码、中央 critic 与 PPO | [PDF](figures/mappo_framework.pdf) · [SVG](figures/mappo_framework.svg) |
| Fig. 3 | 独立测试集上的能耗与可行率 | [PDF](figures/test_performance.pdf) |
| Fig. 4 | N=8 验证集学习曲线 | [PDF](figures/learning_n8.pdf) |
| Fig. 5 | N=8 重叠率与干扰损失诊断 | [PDF](figures/coordination_n8.pdf) |

Fig. 1 的设备位置与到达相位用于解释场景，不是实验实例。Fig. 2 对应 `features.py`、`networks.py`、`policy.py` 与 `trainer.py` 的实际实现；局部乘子仅在 MAPPO-D 中进入 context，空闲期间也更新 GRU。两图均由绘图代码生成，文字与形状可修改，不依赖外部图片。

## 构建

从项目根目录运行：

```bash
make -C manuscript
```

默认使用项目现有 `.venv/bin/python`，需要 NumPy、Matplotlib，以及包含 `IEEEtran` 的 LaTeX 环境、BibTeX 和 `latexmk`。可通过 `PYTHON` 指定另一个已安装依赖的解释器：

```bash
make -C manuscript PYTHON=/absolute/path/to/python
```

`make` 读取脚本中列明的六份归档 JSON，重建统计图表与 `data/evidence.json`，绘制两张模型示意图，再生成 `main.pdf`。源 JSON 有变化时，证据快照的哈希也随之变化；它不是独立实验运行记录。训练日志和模型检查点不参与稿件构建。

若只修改正文，可保留现成图表，直接编译：

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

编译中间文件放在 `build/`，不纳入版本控制。全部正文、图表和参考文献已保存在本目录；脱离仓库时仍可用上述 LaTeX 命令编译，但重建图表需要原仓库中的归档 JSON。

## 初稿状态与投稿前工作

- **格式**：补入场景图与网络图后，当前为标准 `IEEEtran`、10 pt、Letter、双栏的 7 页工作稿，保留既有实验分析，未压缩字号或页边距。已成功编译并检查页面、交叉引用和字体嵌入。目标 ICC 年份、symposium、页数限制和匿名要求尚未最终核实；当前页数不能视为已通过当届投稿合规检查。
- **作者**：暂用匿名占位。正式作者、单位、基金及致谢需由作者补入。
- **数据**：正式结果来自已经读取并完成分析的 EXP-004 测试集；本次写作没有根据测试结果修改方法、超参数或选模规则。
- **最有价值的补充**：优先接入 EXP-005 的完整、按其协议完成的文献学习基线比较；若后续还有实验预算，再选一项能够直接区分收益来源的受控消融。当前稿件正文完整，不把尚未完成的结果写成结论。
- **表述边界**：中央参照要求零违约；`F ≥ 0.95` 是有限样本均值判据；行为诊断不是因果消融；只计算发射能量。具体说明见 [evidence.md](evidence.md)。

## 文献

相关工作覆盖经典节能分组调度、分布式频谱接入、V2X 资源分配、周期任务的可靠性功控、多小区循环 MARL、最接近的 XR-SBPC、MAPPO、状态增强约束 RL 和 Deep Sets。原始论文或作者/机构来源列在 [evidence.md](evidence.md)。参考文献 [8] 使用已查阅的 arXiv 版本；投稿前可以统一替换为经核实的正式出版版本。
