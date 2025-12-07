"""
改进的用户轨迹生成器 - 对角线探索版本
用户沿对角线大方向移动，但在周围区域自由探索
"""
import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
from typing import Dict


class DiagonalExplorationTrajectoryGenerator:
    """对角线探索轨迹生成器 - 大方向对角线，有探索自由度"""
    
    def __init__(
        self, 
        num_users: int = 6,
        num_steps: int = 42,
        area_length: float = 400.0,
        area_width: float = 400.0,
        user_max_speed: float = 2.0,
        time_step: float = 5.0,
        min_task_size: float = 2.5,
        max_task_size: float = 5.0,
        movement_mode: str = 'diagonal_explore',
        diagonal_bias: float = 0.4,  # 🆕 降低对角线偏好，增加探索
        exploration_range: float = 0.35,  # 🆕 探索范围（相对于区域大小）
        inertia_strength: float = 0.7,
        seed: int = None
    ):
        """
        Args:
            diagonal_bias: 对角线移动偏好 [0, 1]
                - 0.2: 很弱的对角线倾向，强探索
                - 0.4: 适中对角线倾向（推荐）✅
                - 0.6: 较强对角线倾向，弱探索
            exploration_range: 探索范围 [0, 1]
                - 0.2: 小范围探索
                - 0.35: 中等范围探索（推荐）✅
                - 0.5: 大范围探索
        """
        self.num_users = num_users
        self.num_steps = num_steps
        self.area_length = area_length
        self.area_width = area_width
        self.user_max_speed = user_max_speed
        self.time_step = time_step
        self.min_task_size = min_task_size
        self.max_task_size = max_task_size
        self.movement_mode = movement_mode
        self.diagonal_bias = diagonal_bias
        self.exploration_range = exploration_range
        self.inertia_strength = inertia_strength
        
        self.group_division = self._divide_users_into_groups()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"✓ 对角线探索轨迹配置:")
        print(f"  - 区域: {area_length}m × {area_width}m")
        print(f"  - 对角线偏好: {diagonal_bias} (越小探索性越强)")
        print(f"  - 探索范围: {exploration_range} (相对比例)")
        print(f"  - 惯性强度: {inertia_strength}")
        print(f"  - 分组: 组A({len(self.group_division['A'])}人)↙, "
              f"组B({len(self.group_division['B'])}人)↗")
    
    def _divide_users_into_groups(self) -> Dict[str, list]:
        """将用户分为两组"""
        groups = {'A': [], 'B': []}
        for user_id in range(self.num_users):
            if user_id % 2 == 0:
                groups['A'].append(user_id)
            else:
                groups['B'].append(user_id)
        return groups
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [0, 2π]"""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """计算两个角度之间的最小差值"""
        diff = angle2 - angle1
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def _smooth_angle_transition(self, current_angle: float, target_angle: float, 
                                 smoothness: float) -> float:
        """平滑角度过渡"""
        angle_diff = self._angle_difference(current_angle, target_angle)
        max_turn = math.pi / 6 * (1 - smoothness)
        
        if abs(angle_diff) > max_turn:
            angle_diff = max_turn if angle_diff > 0 else -max_turn
        
        new_angle = current_angle + angle_diff * (1 - smoothness)
        return self._normalize_angle(new_angle)
    
    def _blend_angles(self, angle1: float, angle2: float, weight: float) -> float:
        """平滑混合两个角度"""
        x1 = math.cos(angle1)
        y1 = math.sin(angle1)
        x2 = math.cos(angle2)
        y2 = math.sin(angle2)
        
        x = (1 - weight) * x1 + weight * x2
        y = (1 - weight) * y1 + weight * y2
        
        blended_angle = math.atan2(y, x)
        return self._normalize_angle(blended_angle)
    
    def _generate_starting_point(self, user_id: int, group: str) -> tuple:
        """生成起点"""
        if group == 'A':
            # 组A: 右上角区域
            x = random.uniform(0.70 * self.area_length, 0.95 * self.area_length)
            y = random.uniform(0.70 * self.area_width, 0.95 * self.area_width)
        else:
            # 组B: 左下角区域
            x = random.uniform(0.05 * self.area_length, 0.30 * self.area_length)
            y = random.uniform(0.05 * self.area_width, 0.30 * self.area_width)
        
        return x, y
    
    def _get_diagonal_direction(self, group: str) -> float:
        """获取对角线方向"""
        if group == 'A':
            return math.pi * 5 / 4  # 225° (左下)
        else:
            return math.pi / 4  # 45° (右上)
    
    def _get_progress_on_diagonal(self, x: float, y: float, group: str) -> float:
        """
        🆕 计算在对角线上的进度 [0, 1]
        
        Args:
            x, y: 当前位置
            group: 用户组
        
        Returns:
            progress: 0=起点，1=终点
        """
        if group == 'A':
            # 组A: 从右上(1,1)到左下(0,0)
            # 进度 = 1 - 平均归一化坐标
            progress = 1 - ((x / self.area_length + y / self.area_width) / 2)
        else:
            # 组B: 从左下(0,0)到右上(1,1)
            # 进度 = 平均归一化坐标
            progress = (x / self.area_length + y / self.area_width) / 2
        
        return max(0, min(1, progress))
    
    def generate_trajectories(self) -> Dict[int, np.ndarray]:
        """生成所有用户的轨迹"""
        trajectories = {}
        
        print(f"\n生成轨迹:")
        
        for group_name, user_ids in self.group_division.items():
            direction_desc = "右上→左下" if group_name == 'A' else "左下→右上"
            print(f"  组{group_name} ({direction_desc}): 用户 {user_ids}")
            
            for user_id in user_ids:
                start_x, start_y = self._generate_starting_point(user_id, group_name)
                
                if self.movement_mode == 'diagonal_explore':
                    trajectory = self._generate_diagonal_explore_walk(
                        user_id, start_x, start_y, group_name
                    )
                elif self.movement_mode == 'diagonal_wander':
                    trajectory = self._generate_diagonal_wander_walk(
                        user_id, start_x, start_y, group_name
                    )
                elif self.movement_mode == 'diagonal_sweep':
                    trajectory = self._generate_diagonal_sweep_walk(
                        user_id, start_x, start_y, group_name
                    )
                else:
                    raise ValueError(f"Unknown movement_mode: {self.movement_mode}")
                
                trajectories[user_id] = trajectory
        
        return trajectories
    
    def _generate_diagonal_explore_walk(self, user_id: int, start_x: float, start_y: float, 
                                        group: str) -> np.ndarray:
        """
        🆕 模式1：对角线探索（推荐）
        - 大方向沿对角线
        - 在对角线周围大范围探索
        - 不会贴着对角线走
        """
        trajectory = np.zeros((self.num_steps, 3))
        
        x, y = start_x, start_y
        max_distance = self.user_max_speed * self.time_step
        
        diagonal_direction = self._get_diagonal_direction(group)
        current_velocity_angle = diagonal_direction
        
        # 🔑 探索参数
        exploration_phase = 0  # 探索相位
        local_target_offset = random.uniform(-math.pi, math.pi)  # 局部目标偏移
        steps_since_target_change = 0
        target_change_interval = random.randint(5, 10)  # 每5-10步改变探索目标
        
        for step in range(self.num_steps):
            trajectory[step] = [x, y, random.uniform(self.min_task_size, self.max_task_size)]
            
            # 计算当前在对角线上的进度
            progress = self._get_progress_on_diagonal(x, y, group)
            
            # 🔑 周期性改变探索目标
            steps_since_target_change += 1
            if steps_since_target_change >= target_change_interval:
                steps_since_target_change = 0
                target_change_interval = random.randint(5, 10)
                # 在对角线方向的±90°范围内选择新的探索方向
                local_target_offset = random.uniform(-math.pi/2, math.pi/2)
                print(f"    用户 {user_id} 在步 {step} 改变探索方向 (偏移={math.degrees(local_target_offset):.0f}°)")
            
            # === 计算期望方向 ===
            
            # 1. 对角线分量（基础方向）
            diagonal_component = diagonal_direction
            
            # 2. 探索分量（局部目标）
            exploration_direction = diagonal_direction + local_target_offset
            exploration_direction = self._normalize_angle(exploration_direction)
            
            # 3. 混合对角线和探索方向
            # 🔑 diagonal_bias 控制偏好：越小，探索性越强
            desired_direction = self._blend_angles(
                exploration_direction,  # 探索方向
                diagonal_component,     # 对角线方向
                self.diagonal_bias      # 对角线权重
            )
            
            # 4. 添加随机漫游（增加不可预测性）
            random_drift = random.uniform(-math.pi / 6, math.pi / 6)
            desired_direction += random_drift * (1 - self.diagonal_bias)
            desired_direction = self._normalize_angle(desired_direction)
            
            # 🔑 应用惯性平滑
            desired_direction = self._smooth_angle_transition(
                current_velocity_angle,
                desired_direction,
                self.inertia_strength
            )
            
            # === 边界处理 ===
            margin = 30
            correction_strength = 0.0
            correction_direction = desired_direction
            
            # 检查四个边界
            if x < margin:
                correction_direction = 0  # 向右
                correction_strength = 0.6
            elif x > self.area_length - margin:
                correction_direction = math.pi  # 向左
                correction_strength = 0.6
            
            if y < margin:
                correction_direction = math.pi / 2  # 向上
                correction_strength = 0.6
            elif y > self.area_width - margin:
                correction_direction = 3 * math.pi / 2  # 向下
                correction_strength = 0.6
            
            # 应用边界修正
            if correction_strength > 0:
                desired_direction = self._blend_angles(
                    desired_direction,
                    correction_direction,
                    correction_strength
                )
            
            # 更新速度方向
            current_velocity_angle = desired_direction
            
            # 移动
            distance = random.uniform(0.75 * max_distance, max_distance)
            new_x = x + distance * math.cos(current_velocity_angle)
            new_y = y + distance * math.sin(current_velocity_angle)
            
            x = max(0, min(self.area_length, new_x))
            y = max(0, min(self.area_width, new_y))
        
        return trajectory
    
    def _generate_diagonal_wander_walk(self, user_id: int, start_x: float, start_y: float, 
                                       group: str) -> np.ndarray:
        """
        🆕 模式2：对角线漫游
        - 大方向对角线
        - 更强的随机漫游特性
        - 探索范围更大
        """
        trajectory = np.zeros((self.num_steps, 3))
        
        x, y = start_x, start_y
        max_distance = self.user_max_speed * self.time_step
        
        diagonal_direction = self._get_diagonal_direction(group)
        current_velocity_angle = random.uniform(0, 2 * math.pi)  # 随机初始方向
        
        # 漫游角度
        wander_angle = 0
        
        for step in range(self.num_steps):
            trajectory[step] = [x, y, random.uniform(self.min_task_size, self.max_task_size)]
            
            progress = self._get_progress_on_diagonal(x, y, group)
            
            # 🔑 漫游角度缓慢变化
            wander_angle += random.uniform(-math.pi / 6, math.pi / 6)
            wander_angle = np.clip(wander_angle, -math.pi, math.pi)
            
            # 每隔一段时间，让漫游角度向对角线回归
            if step % 8 == 0 and random.random() < 0.4:
                wander_angle *= 0.5
            
            # 混合对角线方向和漫游
            exploration_direction = diagonal_direction + wander_angle
            exploration_direction = self._normalize_angle(exploration_direction)
            
            # 弱对角线偏好，强探索
            desired_direction = self._blend_angles(
                exploration_direction,
                diagonal_direction,
                self.diagonal_bias * 0.7  # 更弱的对角线约束
            )
            
            # 应用惯性
            desired_direction = self._smooth_angle_transition(
                current_velocity_angle,
                desired_direction,
                self.inertia_strength
            )
            
            # 边界处理
            margin = 30
            if x < margin or x > self.area_length - margin or \
               y < margin or y > self.area_width - margin:
                # 接近边界时，向区域中心偏移
                to_center_x = self.area_length / 2 - x
                to_center_y = self.area_width / 2 - y
                to_center_angle = math.atan2(to_center_y, to_center_x)
                desired_direction = self._blend_angles(desired_direction, to_center_angle, 0.4)
            
            current_velocity_angle = desired_direction
            
            distance = random.uniform(0.7 * max_distance, max_distance)
            new_x = x + distance * math.cos(current_velocity_angle)
            new_y = y + distance * math.sin(current_velocity_angle)
            
            x = max(0, min(self.area_length, new_x))
            y = max(0, min(self.area_width, new_y))
        
        return trajectory
    
    def _generate_diagonal_sweep_walk(self, user_id: int, start_x: float, start_y: float, 
                                      group: str) -> np.ndarray:
        """
        🆕 模式3：对角线扫描
        - 像扫帚一样，在对角线方向上横向扫描
        - 覆盖更广的区域
        """
        trajectory = np.zeros((self.num_steps, 3))
        
        x, y = start_x, start_y
        max_distance = self.user_max_speed * self.time_step
        
        diagonal_direction = self._get_diagonal_direction(group)
        current_velocity_angle = diagonal_direction
        
        # 扫描参数
        sweep_phase = 0
        sweep_frequency = random.uniform(0.1, 0.2)
        sweep_amplitude = random.uniform(0.5, 0.8)
        
        for step in range(self.num_steps):
            trajectory[step] = [x, y, random.uniform(self.min_task_size, self.max_task_size)]
            
            sweep_phase += sweep_frequency
            
            # 🔑 横向扫描（垂直于对角线方向）
            perpendicular_angle = diagonal_direction + math.pi / 2
            sweep_offset = math.sin(sweep_phase) * sweep_amplitude * math.pi / 2
            
            # 沿对角线前进 + 横向扫描
            forward_component = diagonal_direction
            lateral_component = perpendicular_angle + sweep_offset
            
            # 混合前进和横向
            desired_direction = self._blend_angles(
                lateral_component,
                forward_component,
                self.diagonal_bias
            )
            
            # 应用惯性
            desired_direction = self._smooth_angle_transition(
                current_velocity_angle,
                desired_direction,
                self.inertia_strength
            )
            
            # 边界处理
            margin = 30
            if x < margin or x > self.area_length - margin or \
               y < margin or y > self.area_width - margin:
                to_center_x = self.area_length / 2 - x
                to_center_y = self.area_width / 2 - y
                to_center_angle = math.atan2(to_center_y, to_center_x)
                desired_direction = self._blend_angles(desired_direction, to_center_angle, 0.5)
            
            current_velocity_angle = desired_direction
            
            distance = random.uniform(0.75 * max_distance, max_distance)
            new_x = x + distance * math.cos(current_velocity_angle)
            new_y = y + distance * math.sin(current_velocity_angle)
            
            x = max(0, min(self.area_length, new_x))
            y = max(0, min(self.area_width, new_y))
        
        return trajectory
    
    def save_trajectories(self, trajectories: Dict, filepath: str = "user_trajectories.json"):
        """保存轨迹到JSON"""
        json_data = {
            'metadata': {
                'num_users': self.num_users,
                'num_steps': self.num_steps,
                'area_length': self.area_length,
                'area_width': self.area_width,
                'user_max_speed': self.user_max_speed,
                'time_step': self.time_step,
                'min_task_size': self.min_task_size,
                'max_task_size': self.max_task_size,
                'movement_mode': self.movement_mode,
                'diagonal_bias': self.diagonal_bias,
                'exploration_range': self.exploration_range,
                'inertia_strength': self.inertia_strength,
                'group_division': self.group_division
            },
            'trajectories': {}
        }
        
        for user_id, traj in trajectories.items():
            json_data['trajectories'][str(user_id)] = traj.tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 轨迹数据已保存: {filepath}")
    
    def plot_trajectories(self, trajectories: Dict, save_path: str = "user_trajectories.png"):
        """绘制并保存轨迹图片"""
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # 绘制对角线参考线（虚线，淡化）
        ax.plot([0, self.area_length], [self.area_width, 0], 
               'k:', linewidth=1.5, alpha=0.2, label='对角线参考')
        ax.plot([0, self.area_length], [0, self.area_width], 
               'k:', linewidth=1.5, alpha=0.2)
        
        # 🔑 绘制探索区域范围（对角线周围的带状区域）
        # 组A对角线带
        diagonal_width = self.exploration_range * self.area_length
        points_A = []
        for t in np.linspace(0, 1, 100):
            x_center = (1 - t) * self.area_length
            y_center = (1 - t) * self.area_width
            # 上边界
            points_A.append([x_center + diagonal_width/2, y_center + diagonal_width/2])
        for t in np.linspace(1, 0, 100):
            x_center = (1 - t) * self.area_length
            y_center = (1 - t) * self.area_width
            # 下边界
            points_A.append([x_center - diagonal_width/2, y_center - diagonal_width/2])
        
        points_A = np.array(points_A)
        ax.fill(points_A[:, 0], points_A[:, 1], color='red', alpha=0.08, label='组A探索区域')
        
        # 组B对角线带
        points_B = []
        for t in np.linspace(0, 1, 100):
            x_center = t * self.area_length
            y_center = t * self.area_width
            points_B.append([x_center - diagonal_width/2, y_center + diagonal_width/2])
        for t in np.linspace(1, 0, 100):
            x_center = t * self.area_length
            y_center = t * self.area_width
            points_B.append([x_center + diagonal_width/2, y_center - diagonal_width/2])
        
        points_B = np.array(points_B)
        ax.fill(points_B[:, 0], points_B[:, 1], color='blue', alpha=0.08, label='组B探索区域')
        
        # 标注起始区域
        rect_A = plt.Rectangle(
            (0.70 * self.area_length, 0.70 * self.area_width),
            0.25 * self.area_length, 0.25 * self.area_width,
            fill=True, facecolor='lightcoral', alpha=0.15,
            edgecolor='red', linewidth=2, linestyle='--',
            label='组A起点'
        )
        ax.add_patch(rect_A)
        
        rect_B = plt.Rectangle(
            (0.05 * self.area_length, 0.05 * self.area_width),
            0.25 * self.area_length, 0.25 * self.area_width,
            fill=True, facecolor='lightblue', alpha=0.15,
            edgecolor='blue', linewidth=2, linestyle='--',
            label='组B起点'
        )
        ax.add_patch(rect_B)
        
        # 绘制用户轨迹
        colors_A = plt.cm.Reds(np.linspace(0.5, 0.95, len(self.group_division['A'])))
        colors_B = plt.cm.Blues(np.linspace(0.5, 0.95, len(self.group_division['B'])))
        
        for user_id, traj in trajectories.items():
            x_coords = traj[:, 0]
            y_coords = traj[:, 1]
            
            if user_id in self.group_division['A']:
                color_idx = self.group_division['A'].index(user_id)
                color = colors_A[color_idx]
                label = f'用户 {user_id} (A组↙)'
            else:
                color_idx = self.group_division['B'].index(user_id)
                color = colors_B[color_idx]
                label = f'用户 {user_id} (B组↗)'
            
            # 主轨迹线
            ax.plot(x_coords, y_coords, '-', color=color, 
                   linewidth=4, alpha=0.8, label=label, zorder=5)
            
            # 起点
            ax.plot(x_coords[0], y_coords[0], 'o', color=color, 
                   markersize=20, markeredgecolor='green', markeredgewidth=5, zorder=10)
            
            # 终点
            ax.plot(x_coords[-1], y_coords[-1], 's', color=color, 
                   markersize=20, markeredgecolor='darkred', markeredgewidth=5, zorder=10)
        
        ax.set_xlim(0, self.area_length)
        ax.set_ylim(0, self.area_width)
        ax.set_xlabel('X坐标 (米)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Y坐标 (米)', fontsize=18, fontweight='bold')
        ax.set_title(f'对角线探索轨迹 (偏好={self.diagonal_bias}, 探索={self.exploration_range})\n'
                    f'{self.num_users}个用户 × {self.num_steps}步 | 模式: {self.movement_mode}', 
                    fontsize=20, fontweight='bold', pad=25)
        
        # 图例分两列
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:12], labels[:12], loc='upper left', bbox_to_anchor=(1.02, 1), 
                 fontsize=12, framealpha=0.9, ncol=1)
        
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 轨迹图片已保存: {save_path}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("对角线探索轨迹生成器")
    print("="*70)
    
    generator = DiagonalExplorationTrajectoryGenerator(
        num_users=5,
        num_steps=30,
        area_length=400,
        area_width=400,
        user_max_speed=2.0,
        time_step=5.0,
        min_task_size=15,
        max_task_size=20,
        #movement_mode='diagonal_explore',  # 🔑 推荐！
        # movement_mode='diagonal_wander',   # 或漫游模式
        movement_mode='diagonal_sweep',    # 或扫描模式
        diagonal_bias=0.1,  # 🔑 对角线偏好（越小探索越强）
        exploration_range=0.95,  # 🔑 探索范围
        inertia_strength=0.7,
    )
    
    print("\n正在生成对角线探索轨迹...")
    trajectories = generator.generate_trajectories()
    
    # 验证探索特性
    print("\n验证轨迹特性:")
    for user_id, traj in trajectories.items():
        start_x, start_y = traj[0, 0], traj[0, 1]
        end_x, end_y = traj[-1, 0], traj[-1, 1]
        
        # 计算到对角线的平均距离
        distances_to_diagonal = []
        for i in range(len(traj)):
            x, y = traj[i, 0], traj[i, 1]
            if user_id in generator.group_division['A']:
                # 组A对角线: y = -x + area_length
                dist = abs(y - (-x + generator.area_length)) / math.sqrt(2)
            else:
                # 组B对角线: y = x
                dist = abs(y - x) / math.sqrt(2)
            distances_to_diagonal.append(dist)
        
        avg_dist = np.mean(distances_to_diagonal)
        max_dist = np.max(distances_to_diagonal)
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        if user_id in generator.group_division['A']:
            group = 'A (↙)'
        else:
            group = 'B (↗)'
        
        print(f"  用户 {user_id} ({group}): "
              f"位移=({dx:+.0f}, {dy:+.0f}), "
              f"平均离对角线={avg_dist:.0f}m, 最大={max_dist:.0f}m")
    
    generator.save_trajectories(trajectories, "user_trajectories_hot.json")
    generator.plot_trajectories(trajectories, "user_trajectories_hot.png")
    
    print("\n✅ 完成！用户沿对角线大方向移动，但在周围区域自由探索！")