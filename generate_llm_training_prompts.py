# generate_llm_training_prompts.py
"""
生成完整的LLM训练提示词
将DDPG专家数据 + 系统建模 + 任务描述整合为LLM可学习的完整提示词
"""
import json
import os


class LLMPromptGenerator:
    """LLM训练提示词生成器"""
    
    def __init__(self, ddpg_data_path='llm_training_data/ddpg_llm_training_prompts.json'):
        """
        初始化
        
        Args:
            ddpg_data_path: DDPG专家数据路径
        """
        self.ddpg_data_path = ddpg_data_path
        self.training_data = []
        
        # 加载DDPG专家数据
        if os.path.exists(ddpg_data_path):
            with open(ddpg_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.training_data = data.get('training_data', [])
            print(f"✓ 已加载 {len(self.training_data)} 个DDPG专家样本")
        else:
            print(f"⚠️  数据文件不存在: {ddpg_data_path}")
    
    def get_system_prompt(self):
        """
        获取系统建模提示词
        包含完整的系统配置、物理模型、优化目标等
        """
        system_prompt = """你是一个专业的UAV边缘计算系统智能调度器。你的任务是学习如何优化多无人机边缘计算系统的资源分配和调度策略。

# 系统配置

## 1. 环境与实体

### 区域
- 400×400米的2D平面
- UAV飞行在固定高度50米

### UAV配置（2架）
- **标识**: UAV_0, UAV_1
- **初始位置**: UAV_0在(0, 400), UAV_1在(400, 0)
- **飞行参数**:
  - 最大速度: 20 m/s
  - 时间步长: 1秒
  - 每步最大移动距离: 20米
- **计算能力**:
  - CPU频率: 10 GHz (10×10⁹ cycles/s)
  - 能效系数 κ_uav: 1×10⁻²⁸ J·s²/cycle³
- **通信能力**:
  - 总带宽: 10 MHz
  - 发射功率: 0.2 W
  - 带宽分配: 服务的用户数平分

### 用户配置（5个）
- **标识**: User_0 到 User_4
- **移动模型**: 随机游走，每步随机方向移动
- **计算能力**:
  - CPU频率: 2 GHz (2×10⁹ cycles/s)
  - 能效系数 κ_user: 1×10⁻²⁸ J·s²/cycle³
- **任务特性**:
  - 任务大小: 15-20 Mbits（动态随机生成）
  - 计算密度: 1000 cycles/bit

## 2. 物理模型

### 通信模型
**路径损耗**:
```
L(d) = β₀ · d^(-α)
```
- β₀ = 1×10⁻⁵ (参考路径损耗)
- α = 2.8 (路径损耗指数)
- d = √(d_2D² + h²) (3D距离，h=50m)

**信道容量** (Shannon公式):
```
R = B · log₂(1 + SNR)
SNR = (P_tx · L(d)) / N₀
```
- B: 分配给用户的带宽 (Hz)
- P_tx = 0.2 W (发射功率)
- N₀ = 1×10⁻¹³ W (噪声功率)

### 计算模型
**本地计算**:
```
时延: T_local = (D × (1-ρ) × C) / f_user
能耗: E_local = κ_user × (D × (1-ρ) × C) × f_user²
```

**UAV计算**:
```
时延: T_uav = (D × ρ × C) / f_allocated
能耗: E_uav = κ_uav × (D × ρ × C) × f_allocated²
```
- D: 任务大小 (bits)
- ρ: 卸载比例 [0, 1]
- C: 计算密度 (1000 cycles/bit)
- f_allocated: 分配给用户的UAV频率 (f_uav / 服务用户数)

**传输**:
```
时延: T_trans = D × ρ / R
能耗: E_trans = P_tx × T_trans
```

**总时延** (并行处理):
```
T_total = max(T_local, T_trans + T_uav)
```

## 3. 优化目标

**优化目标**
**最小化整个回合的累积能耗和累积时延**。
- **注意**: 请忽略环境返回的奖励函数数值，专注于物理指标（时延和能耗）的最小化。
- 你的目标是在保证任务完成的前提下，尽可能降低系统的总能耗和总时延。

### 关键权衡
1. **卸载率 vs 性能**:
   - 高卸载 → 降低本地计算时延 → 增加传输能耗和UAV计算能耗
   - 低卸载 → 增加本地计算时延 → 降低传输能耗

2. **UAV负载 vs 能效**:
   - 服务用户少 → f_allocated高 → 计算快但能耗高 (f²效应)
   - 服务用户多 → f_allocated低 → 计算慢但能耗低

3. **距离 vs 传输**:
   - 距离近 → SNR高 → 传输快、能耗低 → 适合高卸载
   - 距离远 → SNR低 → 传输慢、能耗高 → 适合低卸载

## 4. 决策变量

### 用户分配 (user_assignments)
```json
{"0": 0, "1": 1, "2": 0, "3": 1, "4": 0}
```
- 键: 用户ID (0-4)
- 值: UAV ID (0或1)
- 约束: 每个用户必须且只能连接一个UAV

### 卸载比例 (offloading_ratios)
```json
{"0": 0.7, "1": 0.6, "2": 0.8, "3": 0.5, "4": 0.7}
```
- 键: 用户ID (0-4)
- 值: 卸载比例 ρ ∈ [0, 1]
- 0 = 全本地计算, 1 = 全卸载

### UAV移动 (uav_movements)
```json
{"0": {"dx": 5.0, "dy": 0.0}, "1": {"dx": -5.0, "dy": 5.0}}
```
- 键: UAV ID (0-1)
- 值: 位移向量 (dx, dy)
- 约束: dx, dy ∈ [-20, 20] 米

## 5. 决策策略指导

### 负载均衡原则
- 目标: 让两个UAV服务的用户数接近 (如2 vs 3)
- 原因: 避免单UAV过载 → 时延爆炸; 避免单UAV空闲 → 算力浪费

### 位置优化原则
- 策略: UAV应向其服务的用户群中心移动
- 效果: 缩短通信距离 → 提升SNR → 降低传输时延和能耗

### 卸载策略原则
**高卸载场景** (ρ > 0.7):
- 用户距离UAV近 (SNR高)
- UAV负载低 (分配频率高)
- 任务量大 (本地计算慢)

**低卸载场景** (ρ < 0.3):
- 用户距离UAV远 (SNR低)
- UAV负载高 (分配频率低)
- 任务量小 (本地计算快)

**混合策略** (0.3 ≤ ρ ≤ 0.7):
- 距离和负载适中
- 平衡本地和UAV计算

### 能耗权衡
**关键洞察**: E ∝ f²
- UAV服务1个用户: f = 10GHz → 快但能耗极高
- UAV服务5个用户: f = 2GHz → 慢但能耗可控
- **最优**: 2-3个用户/UAV，平衡速度和能耗

---

# 学习任务

现在，你将通过学习一系列由训练好的深度强化学习模型（DDPG）生成的专家决策示例，来掌握在不同场景下的最优调度策略。

**学习目标**:
1. 理解状态（UAV位置、用户位置、任务大小）与决策之间的关系
2. 学会在负载均衡、距离优化、能效权衡之间做出权衡
3. 掌握根据具体场景灵活调整卸载率和UAV移动的能力

**注意**: 这些示例来自经过大量训练优化的模型，代表了在当前权重配置(w_delay=0.4, w_energy=0.6)下的近似最优策略。
"""
        return system_prompt
    
    def generate_few_shot_prompts(self, num_examples=10, output_file='llm_few_shot_training.txt'):
        """
        生成few-shot学习格式的完整提示词
        
        Args:
            num_examples: 使用的示例数量
            output_file: 输出文件名
        """
        if not self.training_data:
            print("⚠️  没有训练数据可用")
            return
        
        # 选择样本（可以按奖励排序或随机选择）
        samples = sorted(self.training_data, 
                        key=lambda x: x['outcome']['reward'], 
                        reverse=True)[:num_examples]
        
        # 构建完整提示词
        full_prompt = self.get_system_prompt()
        full_prompt += "\n\n" + "="*70 + "\n"
        full_prompt += "# DDPG专家决策示例\n"
        full_prompt += "="*70 + "\n\n"
        
        for idx, sample in enumerate(samples, 1):
            full_prompt += f"\n## 示例 {idx}\n\n"
            full_prompt += "### 输入场景:\n```\n"
            full_prompt += sample['input']
            full_prompt += "\n```\n\n"
            full_prompt += "### DDPG专家决策:\n```json\n"
            full_prompt += sample['output']
            full_prompt += "\n```\n\n"
            full_prompt += "### 决策效果:\n"
            full_prompt += f"- 奖励: {sample['outcome']['reward']:.4f}\n"
            full_prompt += f"- 平均时延: {sample['outcome']['avg_delay']:.4f}s\n"
            full_prompt += f"- 总能耗: {sample['outcome']['total_energy']:.2f}J\n"
            
            # 添加决策分析
            analysis = self._analyze_decision(sample)
            full_prompt += "\n### 决策分析:\n"
            full_prompt += analysis
            full_prompt += "\n" + "-"*70 + "\n"
        
        # 保存
        output_path = os.path.join(os.path.dirname(self.ddpg_data_path), output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        
        print(f"✓ Few-shot训练提示词已保存: {output_path}")
        print(f"  包含 {len(samples)} 个专家示例")
        return output_path
    
    def generate_fine_tuning_dataset(self, output_file='llm_finetuning_dataset.jsonl'):
        """
        生成fine-tuning格式的数据集 (JSONL格式)
        每行一个训练样本，格式: {"messages": [{"role": "system"...}, {"role": "user"...}, {"role": "assistant"...}]}
        """
        system_prompt = self.get_system_prompt()
        
        output_path = os.path.join(os.path.dirname(self.ddpg_data_path), output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.training_data:
                training_sample = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sample['input']},
                        {"role": "assistant", "content": sample['output']}
                    ]
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
        
        print(f"✓ Fine-tuning数据集已保存: {output_path}")
        print(f"  包含 {len(self.training_data)} 个训练样本")
        print(f"  格式: JSONL (适用于OpenAI/Claude fine-tuning)")
        return output_path
    
    def _analyze_decision(self, sample):
        """分析决策的关键特征"""
        decision = json.loads(sample['output'])
        
        analysis = ""
        
        # 1. 负载分布
        user_assigns = decision['user_assignments']
        uav0_count = sum(1 for v in user_assigns.values() if int(v) == 0)
        uav1_count = sum(1 for v in user_assigns.values() if int(v) == 1)
        analysis += f"**负载分布**: UAV_0服务{uav0_count}个用户, UAV_1服务{uav1_count}个用户"
        if abs(uav0_count - uav1_count) <= 1:
            analysis += " ✓ (良好均衡)\n"
        else:
            analysis += " (负载不均)\n"
        
        # 2. 平均卸载比例
        ratios = [float(v) for v in decision['offloading_ratios'].values()]
        avg_offload = sum(ratios) / len(ratios)
        analysis += f"**平均卸载率**: {avg_offload:.2f}"
        if avg_offload > 0.7:
            analysis += " (高度依赖UAV计算)\n"
        elif avg_offload < 0.3:
            analysis += " (主要本地计算)\n"
        else:
            analysis += " (混合策略)\n"
        
        # 3. UAV移动
        movements = decision['uav_movements']
        total_movement = sum(abs(m['dx']) + abs(m['dy']) for m in movements.values())
        analysis += f"**UAV总移动距离**: {total_movement:.1f}m"
        if total_movement < 5:
            analysis += " (保持位置)\n"
        elif total_movement > 30:
            analysis += " (大幅调整)\n"
        else:
            analysis += " (适度移动)\n"
        
        return analysis
    
    def get_statistics(self):
        """获取数据统计"""
        if not self.training_data:
            return {}
        
        rewards = [s['outcome']['reward'] for s in self.training_data]
        delays = [s['outcome']['avg_delay'] for s in self.training_data]
        energies = [s['outcome']['total_energy'] for s in self.training_data]
        
        return {
            'num_samples': len(self.training_data),
            'reward': {
                'mean': sum(rewards) / len(rewards),
                'min': min(rewards),
                'max': max(rewards)
            },
            'avg_delay': {
                'mean': sum(delays) / len(delays),
                'min': min(delays),
                'max': max(delays)
            },
            'total_energy': {
                'mean': sum(energies) / len(energies),
                'min': min(energies),
                'max': max(energies)
            }
        }


def main():
    """主函数"""
    print("="*70)
    print("LLM训练提示词生成器")
    print("="*70)
    
    # 创建生成器
    generator = LLMPromptGenerator(
        ddpg_data_path='llm_training_data/ddpg_llm_training_prompts.json'
    )
    
    # 打印统计
    stats = generator.get_statistics()
    if stats:
        print(f"\n数据统计:")
        print(f"  样本数: {stats['num_samples']}")
        print(f"  平均奖励: {stats['reward']['mean']:.4f}")
        print(f"  平均时延: {stats['avg_delay']['mean']:.4f}s")
        print(f"  平均能耗: {stats['total_energy']['mean']:.2f}J")
    
    # 生成few-shot学习格式
    print("\n[1/2] 生成Few-shot学习提示词...")
    generator.generate_few_shot_prompts(
        num_examples=15,
        output_file='llm_few_shot_training.txt'
    )
    
    # 生成fine-tuning数据集
    print("\n[2/2] 生成Fine-tuning数据集...")
    generator.generate_fine_tuning_dataset(
        output_file='llm_finetuning_dataset.jsonl'
    )
    
    print("\n" + "="*70)
    print("✓ 完成！生成的文件:")
    print("  1. llm_few_shot_training.txt - Few-shot学习格式（可直接输入LLM）")
    print("  2. llm_finetuning_dataset.jsonl - Fine-tuning数据集（JSONL格式）")
    print("="*70)


if __name__ == "__main__":
    main()