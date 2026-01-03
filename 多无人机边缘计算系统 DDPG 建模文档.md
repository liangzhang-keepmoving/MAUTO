# 系统建模与 MDP 设计文档

本文档系统梳理了训练环境中的 **马尔可夫决策过程（MDP）建模**、**通信-计算-移动联合建模** 及 **奖励函数设计**，适用于基于强化学习的 UAV 边缘计算任务卸载与轨迹优化研究。

> **注**：所有代码引用路径基于项目根目录 `d:\code\1226\`。

---

## 1. MDP 建模（训练环境）

### 1.1 时间与 episode 结构
- **时间粒度**：每个 `step` 对应物理时间 `Δt = time_step = 1.0 s`。
- **最大步数**：每个 episode 最多运行 `T_max = target_episode_steps = 30` 步（[env_train.py#L24-L58](file:///d:/code/1226/env_train.py#L24-L58)）。
- **终止条件**：达到 `T_max` 或累计完成任务量达标（实践中等价于固定长度截断）（[env_train.py#L242-L259](file:///d:/code/1226/env_train.py#L242-L259)）。

### 1.2 状态空间 $s_t$
神经网络输入为归一化字典，包含以下字段：

| 字段 | 维度 | 含义 | 归一化方式 |
|------|------|------|------------|
| `uav_pos` | $[N, 2]$ | UAV 平面坐标 $(x, y)$ | $x / L$, $y / W$ |
| `user_pos` | $[M, 2]$ | 用户平面坐标 | $x / L$, $y / W$ |
| `user_tasks` | $[M, 1]$ | 当前任务大小（Mbits） | $D_i / D_{\max}$ |

其中 $L = \texttt{area\_length}$, $W = \texttt{area\_width}$, $D_{\max} = \texttt{max\_task\_size}$（[env_train.py#L262-L300](file:///d:/code/1226/env_train.py#L262-L300)）。

### 1.3 动作空间 $a_t$
动作以字典形式组织，按 UAV 输出：

| 字段 | 维度 | 含义 | 备注 |
|------|------|------|------|
| `user_competition_probs` | $[N, M]$ | 每个 UAV 对各用户的“竞争概率” | 用于硬分配（argmax） |
| `offloading_ratios` | $[N, M]$ 或 $[M]$ | 卸载比例 $\rho_i \in [0,1]$ | 训练中广播至所有 UAV |
| `move_vector` | $[N, 2]$ | 位移向量 $(dx, dy)$（单位：米） | 受最大飞行距离约束 |

（[env_train.py#L334-L340](file:///d:/code/1226/env_train.py#L334-L340)，[train_ddpg.py#L155-L181](file:///d:/code/1226/train_ddpg.py#L155-L181)）

### 1.4 状态转移流程（Step 执行）
给定 $(s_t, a_t)$，环境执行以下顺序：

1. **用户分配**  
   对每个用户 $i$，取所有 UAV 的 `user_competition_probs[:, i]` 的 `argmax` 得到分配结果 `user_assignments[i] ∈ {1,…,N}`（[user_allocation.py#L22-L66](file:///d:/code/1226/user_allocation.py#L22-L66)）。

2. **UAV 移动**  
   - 更新位置：$\mathbf{p}_u^{(t+1)} = \text{clip}(\mathbf{p}_u^{(t)} + \Delta \mathbf{p}_u, \mathcal{B})$  
   - 计算移动能耗（见 §4）  
   - 计算越界惩罚（见 §5）  
   （[uav_movement.py#L63-L134](file:///d:/code/1226/uav_movement.py#L63-L134)）

3. **任务处理**  
   基于分配结果与卸载比例，调用 `Calculator` 计算：
   - 传输时延 $T^{\text{tx}}_i$
   - 本地/卸载计算时延 $T^{\text{local}}_i, T^{\text{uav}}_i$
   - 各类能耗（见 §3 & §4）  
   （[calculator.py#L27-L153](file:///d:/code/1226/calculator.py#L27-L153)）

4. **奖励计算**  
   聚合性能指标并加权求和（见 §6）（[reward_system.py#L58-L228](file:///d:/code/1226/reward_system.py#L58-L228)）。

5. **用户演化**  
   更新用户位置（预设轨迹或随机游走）及下一时刻任务大小（[env_train.py#L178-L198](file:///d:/code/1226/env_train.py#L178-L198)）。

---

## 2. 信道与传输建模

### 2.1 几何与路径损耗
- **UAV 高度**：固定 $h = \texttt{uav\_height} = 50\,\text{m}$。
- **3D 距离**：
  $$
  d_{i,u} = \sqrt{(x_u - x_i)^2 + (y_u - y_i)^2 + h^2}
  $$
  （[calculator.py#L208-L270](file:///d:/code/1226/calculator.py#L208-L270)）

- **路径损耗模型（简化幂律）**：
  $$
  L(d) = \frac{\beta_0}{d^\alpha}, \quad \beta_0 = \texttt{reference\_path\_loss} = 10^{-5},\ \alpha = \texttt{path\_loss\_exponent} = 2
  $$
  （[env_train.py#L40-L43](file:///d:/code/1226/env_train.py#L40-L43)）

### 2.2 传输速率与时延
- **接收功率**：$P_{\text{rx}} = P_{\text{tx}} \cdot L(d)$
- **信噪比**：$\text{SNR} = P_{\text{rx}} / N_0$，其中 $N_0 = \texttt{noise\_power}$
- **香农速率**：
  $$
  R = B \log_2(1 + \text{SNR}), \quad B = \texttt{uav\_bandwidth\_per\_user} \times 10^6\ \text{Hz}
  $$
  （[calculator.py#L218-L235](file:///d:/code/1226/calculator.py#L218-L235)）

- **传输时延与能耗**（任务大小 $D_i$ 单位为 Mbits → bits）：
  $$
  T^{\text{tx}}_i = \frac{D_i \cdot 10^6 \cdot \rho_i}{R}, \quad E^{\text{tx}}_i = P_{\text{tx}} \cdot T^{\text{tx}}_i
  $$
  （[calculator.py#L194-L205](file:///d:/code/1226/calculator.py#L194-L205)）

---

## 3. 时延建模（本地 + 卸载并行）

### 3.1 任务拆分
对用户 $i$，任务大小 $D_i$（Mbits），卸载比例 $\rho_i$：
$$
D^{\text{local}}_i = D_i (1 - \rho_i), \quad D^{\text{off}}_i = D_i \rho_i
$$
（[calculator.py#L66-L127](file:///d:/code/1226/calculator.py#L66-L127)）

### 3.2 本地计算
- **CPU 周期数**：$F^{\text{local}}_i = C \cdot D^{\text{local}}_i \cdot 10^6$，$C = \texttt{cpu\_cycles\_per\_bit}$
- **本地时延**：
  $$
  T^{\text{local}}_i = \frac{F^{\text{local}}_i}{f_{\text{user}}}, \quad f_{\text{user}} = \texttt{user\_cpu\_frequency}
  $$
  （[calculator.py#L155-L173](file:///d:/code/1226/calculator.py#L155-L173)）

### 3.3 UAV 侧计算（资源均分）
若 UAV $u$ 服务 $K_u$ 个用户，则：
$$
f^{\text{per}}_u = \frac{f_{\text{uav}}}{K_u}, \quad B^{\text{per}}_u = \frac{B}{K_u}
$$
（[calculator.py#L50-L60](file:///d:/code/1226/calculator.py#L50-L60)）

- **UAV 计算时延**：
  $$
  T^{\text{uav}}_i = \frac{C \cdot D^{\text{off}}_i \cdot 10^6}{f^{\text{per}}_u}
  $$

### 3.4 用户体验时延
卸载路径为 **串行**（先传后算），但本地与卸载 **并行执行**，故：
$$
T^{\text{actual}}_i = \max\left( T^{\text{local}}_i,\ T^{\text{tx}}_i + T^{\text{uav}}_i \right)
$$
（[calculator.py#L125-L127](file:///d:/code/1226/calculator.py#L125-L127)）

> ✅ **简化假设**：任务在单步内必须完成，**无队列、缓存或跨步排队时延**。

---

## 4. 能耗建模

### 4.1 本地计算能耗（CMOS 模型）
$$
E^{\text{local}}_i = \kappa_{\text{user}} \cdot F^{\text{local}}_i \cdot f_{\text{user}}^2
$$
（[calculator.py#L155-L173](file:///d:/code/1226/calculator.py#L155-L173)）

### 4.2 UAV 计算能耗
$$
E^{\text{uav}}_i = \kappa_{\text{uav}} \cdot (C D^{\text{off}}_i 10^6) \cdot \left(f^{\text{per}}_u\right)^2
$$
（[calculator.py#L238-L256](file:///d:/code/1226/calculator.py#L238-L256)）

### 4.3 传输能耗
$$
E^{\text{tx}}_i = P_{\text{tx}} \cdot T^{\text{tx}}_i
$$
（[calculator.py#L194-L205](file:///d:/code/1226/calculator.py#L194-L205)）

### 4.4 UAV 移动能耗（旋翼机模型，Zeng et al. 2019）
给定位移 $\Delta \mathbf{p}$，速度 $V = \|\Delta \mathbf{p}\| / \Delta t$，总功率：
$$
\begin{aligned}
P(V) =\ & P_0 \left(1 + \frac{3V^2}{U_{\text{tip}}^2}\right) \\
& + P_i \sqrt{ \sqrt{1 + \frac{V^4}{4v_0^4}} - \frac{V^2}{2v_0^2} } \\
& + \frac{1}{2} d_0 \rho s A V^3
\end{aligned}
$$
能耗：$E^{\text{move}} = P(V) \cdot \Delta t$  
（[uav_movement.py#L9-L44](file:///d:/code/1226/uav_movement.py#L9-L44)，[L109-L127](file:///d:/code/1226/uav_movement.py#L109-L127)）

---

## 5. 移动、边界与越界惩罚

### 5.1 最大飞行距离约束
若 $\|\Delta \mathbf{p}\| > D_{\max} = \texttt{max\_flight\_distance}$，则缩放：
$$
\Delta \mathbf{p} \leftarrow \Delta \mathbf{p} \cdot \frac{D_{\max}}{\|\Delta \mathbf{p}\|}
$$
（[uav_movement.py#L83-L91](file:///d:/code/1226/uav_movement.py#L83-L91)）

### 5.2 边界裁剪
新位置限制在矩形区域 $\mathcal{B} = [0, L] \times [0, W]$：
$$
\mathbf{p}_u \leftarrow \text{clip}(\mathbf{p}_u, 0, [L, W])
$$
（[uav_movement.py#L93-L102](file:///d:/code/1226/uav_movement.py#L93-L102)）

### 5.3 越界惩罚（归一化）
理论位移 $d^{\text{theory}} = \|\Delta \mathbf{p}\|$，实际位移 $d^{\text{actual}} = \|\mathbf{p}^{(t+1)} - \mathbf{p}^{(t)}\|$，惩罚项：
$$
\text{penalty} = \min\left( \frac{ \sum_u \max(d^{\text{theory}}_u - d^{\text{actual}}_u, 0) }{ \sum_u d^{\text{theory}}_u },\ 1 \right)
$$
（[uav_movement.py#L73-L133](file:///d:/code/1226/uav_movement.py#L73-L133)）

---

## 6. 奖励函数设计

### 6.1 聚合指标
| 指标 | 含义 | 归一化方式 |
|------|------|------------|
| `avg_delay` | $\frac{1}{M} \sum_i T^{\text{actual}}_i$ | $ / D_{\max}^{\text{norm}}$ |
| `total_task_energy` | $\sum_i (E^{\text{local}}_i + E^{\text{tx}}_i + E^{\text{uav}}_i)$ | $ / E_{\max}^{\text{task}}$ |
| `total_move_energy` | $\sum_u E^{\text{move}}_u$ | $ / E_{\max}^{\text{move}}$ |
| `norm_distance` | 平均用户-UAV 距离归一化 | $d / \sqrt{L^2 + W^2 + h^2}$ |
| `norm_load_imbalance` | 负载方差归一化 | 基于最大可能不均衡 |
| `norm_boundary` | 越界惩罚（0–1） | 无需额外归一化 |

归一化上界来自 `norm_params.json`（[env_train.py#L137-L147](file:///d:/code/1226/env_train.py#L137-L147)；[reward_system.py#L103-L120](file:///d:/code/1226/reward_system.py#L103-L120)）。

### 6.2 最终奖励
$$
r_t = - \left(
w_d \hat{D} +
w_{\text{task}} \widehat{E}_{\text{task}} +
w_{\text{move}} \widehat{E}_{\text{move}} +
w_{\text{dist}} \hat{d} +
w_{\text{load}} \widehat{\text{imb}} +
w_b \hat{b}
\right)
$$
权重 $w_\cdot$ 为超参数（[reward_system.py#L198-L208](file:///d:/code/1226/reward_system.py#L198-L208)）。

---

## 附录：关键简化假设总结

1. **任务原子性**：每步任务必须在当前 step 内完成，**无任务排队或跨步延迟**。
2. **信道静态**：每步内信道不变，忽略快衰落。
3. **UAV 高度固定**：不优化垂直维度。
4. **带宽均分**：同一 UAV 服务的用户平分带宽与 CPU。
5. **理想调度**：用户分配由竞争概率 argmax 决定，无冲突解决机制。

---