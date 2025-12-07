# Task Offloading with LLM-Enhanced Multi-Agent Reinforcement Learning in UAV-Assisted Edge Computing

这篇文章的中文题目为《面向无人机辅助边缘计算的LLM增强多智能体强化学习任务卸载》，发表于《Sensors》期刊（2025年第25卷）。文章提出了一种结合大语言模型（LLM）与多智能体强化学习（MARL）的新框架LLM-QTRAN，用于优化多无人机的轨迹规划与任务卸载策略，主要应用于边缘计算环境，尤其是在资源受限或偏远地区。

---

## 📌 一、研究背景与动机

- 无人机（UAV）作为移动边缘计算（MEC）平台，能为用户设备（UE）提供计算卸载服务。
- 多无人机协同工作中存在局部观测与全局状态不一致的问题，影响任务完成率和收敛速度。
- 现有方法如QMIX、VDN、QTRAN等存在结构限制，难以有效处理复杂环境下的协作任务。

---

## 📌 二、核心贡献

1. **提出LLM-QTRAN算法**：
    - 将QTRAN与LLM结合，实现区域划分与任务特征建模。
    - 引入图卷积网络（GCN）和自注意力机制，提升跨区域依赖建模能力。
2. **框架设计**：
    - 建模为多智能体部分可观测马尔可夫决策过程（Dec-POMDP）。
    - 实现集中训练、分布式执行（CTDE）机制。
3. **性能提升**：
    - 相比QTRAN、QMIX等算法，任务成功率提升超过10%，收敛速度更快。

---

## 📌 三、系统模型

- **网络模型**：多UAV为UE提供计算服务，UAV可返回充电站。
- **通信模型**：采用视距（LoS）信道模型，使用TDMA方式。
- **任务卸载模型**：任务完全卸载至UAV处理，考虑通信与计算时延。
- **优化目标**：最大化任务卸载成功率，受限于UAV覆盖范围、飞行距离、计算能力等。

系统建模参考的文献23-25

---

## 📌 四、算法结构

### 1. MARL建模（Dec-POMDP）

- 状态空间、动作空间、观测空间、奖励函数设计。
- 奖励函数包括任务完成率、路径代价与碰撞惩罚。

### 2. LLM模块

- 利用预训练LLM进行区域划分，依据任务特征、用户分布、UAV位置等。
- 输出区域划分策略，提升任务适配性与资源利用率。

### 3. GCN与自注意力机制

- GCN用于建模区域内UAV之间的空间关系。
- 自注意力机制用于动态评估不同区域间的重要性，提升全局决策能力。

---

## 📌 五、实验结果

- **仿真平台**：Python + MPE（Multi-Agent Particle Environment）
- **对比算法**：QTRAN、Fine-tuned QMIX、K-means QTRAN
- **性能指标**：任务成功率、平均奖励、收敛速度

| 算法 | 平均收敛步数 | 平均成功率 | 备注 |
| --- | --- | --- | --- |
| QTRAN | 0.54×10⁵ | 70.4% | 基准方法 |
| Fine-tuned QMIX | 0.22×10⁵ | 80.2% | 基准方法 |
| K-means QTRAN | 0.25×10⁵ | 83.3% | 基准方法 |
| LLM-QTRAN（提出） | 0.18×10⁵ | 93.6% | 本文提出方法 |

---

## 📌 六、结论

- LLM-QTRAN显著提升了任务成功率与收敛速度，适用于动态、资源受限的边缘计算环境。
- 未来将考虑能耗与通信延迟的联合优化，拓展至更大规模UAV系统。

---

## 📌benchmark算法代码

https://github.com/oxwhirl/pymarl

- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

## 📌参考文献整理

| 引文序号 | 文献（年份） | 一句话作用 | 原文链接 | 开源代码/实现 |
| --- | --- | --- | --- | --- |
| [2] | Ding Y. 等 (2023) | 安全计算卸载与资源管理联合优化 | [IEEE 预印](https://doi.org/10.1109/JSTSP.2023.3285505) | N/A |
| [3] | Liu X. 等 (2019) | 模拟退火轨迹优化，启发式适应性差 | [IEEE IoT-J](https://doi.org/10.1109/JIOT.2019.2898600) | N/A |
| [4] | Haber E. E. 等 (2021) | URLLC 超低时延卸载 | [IEEE TCOMM](https://doi.org/10.1109/TCOMM.2021.3093709) | N/A |
| [5] | Zhang J. 等 (2019) | 随机任务到达下轨迹-卸载比 | [IEEE IoT-J](https://doi.org/10.1109/JIOT.2019.2903260) | N/A |
| [6] | Sunehag P. 等 (2018) | VDN 价值分解起点 | [arXiv:1706.05296](https://arxiv.org/abs/1706.05296) | [PyMARL](https://github.com/oxwhirl/pymarl) |
| [7] | Rashid T. 等 (2020) | QMIX 单调性约束 | [arXiv:1803.11485](https://arxiv.org/abs/1803.11485) | [PyMARL](https://github.com/oxwhirl/pymarl) |
| [8] | Son K. 等 (2019) | QTRAN 解除单调限制 | [PMLR](https://proceedings.mlr.press/v97/son19a.html) | [TF1 代码](https://github.com/swkim965/QTRAN) |
| [9] | Wang J. 等 (2020) | QPLEX 双通道 IGM 变换 | [arXiv:2008.01062](https://arxiv.org/abs/2008.01062) | [QPLEX-GitHub](https://github.com/chengannan/QPLEX) |
| [10] | Pina R. 等 (2024) | RQN 残差分解 | [IEEE T-NNLS](https://doi.org/10.1109/TNNLS.2023.3237008) | N/A |
| [11] | Ding T. 等 (2024) | UMIX 能耗改进 QMIX | [arXiv:2210.12362](https://arxiv.org/abs/2210.12362) | N/A |
| [12] | Tan S. 等 (2023) | 通信辅助 VDN 提升公平性 | [IEEE WCL](https://doi.org/10.1109/LWC.2022.3226701) | N/A |
| [13] | Chen L. 等 (2024) | LLM 降低 RL 样本复杂度 | [IEEE RA-L](https://doi.org/10.1109/LRA.2024.3390156) | N/A |
| [14] | Xu Y. 等 (2024) | 首篇 UAV+LLM 应急网络 | [IEEE IPSN](https://doi.org/10.1109/IPSN54338.2024.00055) | N/A |
| [15] | Han Y. 等 (2024) | LLM 生成 6-DoF 飞控先验 | [IEEE Access](https://doi.org/10.1109/ACCESS.2024.3421890) | N/A |
| [16] | Nascimento N. 等 (2023) | Self-Adaptive LLM-Agent 概念 | [IEEE ACSOS-C](https://doi.org/10.1109/ACSOS-C58294.2023.00031) | N/A |
| [17] | Nascimento N. 等 (2024) | LLM 多模态语义抽取示例 | [IEEE ICWS](https://doi.org/10.1109/ICWS60048.2024.00121) | N/A |
| [18] | Wang J. 等 (2025) | 复杂指令→子任务分解 | [IEEE RA-L](https://doi.org/10.1109/LRA.2024.3456789) | [JACK-GitHub](https://github.com/jiayi-wang/JACK) |
| [19] | Zhang Y. 等 (2020) | 首篇 UAV+GCN 着陆决策 | [IEEE ICARM](https://doi.org/10.1109/ICARM49701.2020.00065) | N/A |
| [20] | Peng H. 等 (2024) | GCN-Transformer 时空故障诊断 | [IEEE TAES](https://doi.org/10.1109/TAES.2023.3237845) | N/A |

## 📌遗留问题

1. 本文是0/1 的offloading，文献调研
    1. 连续的offloading
    2. 具体任务类型