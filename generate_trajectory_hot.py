"""
改进的用户轨迹生成器 - 对角线探索版本（左上→右下）
用户沿对角线大方向移动（从左上角到右下角），但在周围区域自由探索
"""
import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
from typing import Dict


class LeftTopToRightBottomTrajectoryGenerator:
    """对角线探索轨迹生成器 - 从左上到右下"""
    
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
        diagonal_bias: float = 0.4,
        exploration_range: float = 0.35,
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
        
        print(f"✓ 对角线探索轨迹配置 (左上→右下):")
        print(f"  - 区域: {area_length}m × {area_width}m")
        print(f"  - 对角线方向: 左上角 → 右下角 (↘)")
        print(f"  - 对角线偏好: {diagonal_bias} (越小探索性越强)")
        print(f"  - 探索范围: {exploration_range} (相对比例)")
        print(f"  - 惯性强度: {inertia_strength}")
        print(f"  - 分组: 组A({len(self.group_division['A'])}人)左上→右下, "
              f"组B({len(self.group_division['B'])}人)右下→左上")
    
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
        """
        🆕 根据对角线方向生成起点
        组A: 左上角 → 右下角
        组B: 右下角 → 左上角
        """
        if group == 'A':
            # 组A: 左上角区域
            x = random.uniform(0.05 * self.area_length, 0.30 * self.area_length)
            y = random.uniform(0.70 * self.area_width, 0.95 * self.area_width)
        else:
            # 组B: 右下角区域
            x = random.uniform(0.70 * self.area_length, 0.95 * self.area_length)
            y = random.uniform(0.05 * self.area_width, 0.30 * self.area_width)
        
        return x, y
    
    def _get_skew_diagonal_direction(self, group: str) -> float:
        """
        🆕 获取对角线方向（从左上到右下）
        
        Returns:
            angle: 弧度制角度
        """
        if group == 'A':
            # 组A: 从左上往右下，角度 = -45° 或 315° (向右下)
            # 在坐标系中：0° = 右，90° = 上，180° = 左，270° = 下
            # 右下方向 = 315° = -45°
            return math.pi * 7 / 4  # 315°
        else:
            # 组B: 从右下往左上，角度 = 135° (向左上)
            return math.pi * 3 / 4  # 135°
    
    def _get_progress_on_diagonal(self, x: float, y: float, group: str) -> float:
        """
        🆕 计算在对角线上的进度 [0, 1]
        左上(0,1) → 右下(1,0)
        """
        # 归一化坐标
        x_norm = x / self.area_length
        y_norm = y / self.area_width
        
        if group == 'A':
            # 组A: 从左上(0,1)到右下(1,0)
            # 进度 = x增加 + y减少
            progress = (x_norm + (1 - y_norm)) / 2
        else:
            # 组B: 从右下(1,0)到左上(0,1)
            # 进度 = x减少 + y增加
            progress = ((1 - x_norm) + y_norm) / 2
        
        return max(0, min(1, progress))
    
    def _get_diagonal_line_equation(self, group: str) -> tuple:
        """
        🆕 获取对角线的直线方程参数
        左上到右下：y = -x + area_width (斜率=-1)
        
        Returns:
            (slope, intercept): 斜率和截距，方程为 y = slope * x + intercept
        """
        # 从左上到右下的对角线斜率为 -1
        slope = -1.0
        
        if group == 'A':
            # 组A: 通过左上角 (0, area_width)
            intercept = self.area_width
        else:
            # 组B: 通过右下角 (area_length, 0)
            # y = -x + intercept
            # 0 = -area_length + intercept
            intercept = self.area_length
        
        return slope, intercept
    
    def generate_trajectories(self) -> Dict[int, np.ndarray]:
        """生成所有用户的轨迹"""
        trajectories = {}
        
        print(f"\n生成轨迹:")
        
        for group_name, user_ids in self.group_division.items():
            direction_desc = "左上→右下(↘)" if group_name == 'A' else "右下→左上(↖)"
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
        🆕 模式1：斜对角线探索（推荐）
        - 大方向沿斜对角线
        - 在斜对角线周围大范围探索
        """
        trajectory = np.zeros((self.num_steps, 3))
        
        x, y = start_x, start_y
        max_distance = self.user_max_speed * self.time_step
        
        diagonal_direction = self._get_skew_diagonal_direction(group)
        current_velocity_angle = diagonal_direction
        
        # 探索参数
        exploration_phase = 0
        local_target_offset = random.uniform(-math.pi, math.pi)
        steps_since_target_change = 0
        target_change_interval = random.randint(5, 10)
        
        for step in range(self.num_steps):
            trajectory[step] = [x, y, random.uniform(self.min_task_size, self.max_task_size)]
            
            progress = self._get_progress_on_diagonal(x, y, group)
            
            # 周期性改变探索目标
            steps_since_target_change += 1
            if steps_since_target_change >= target_change_interval:
                steps_since_target_change = 0
                target_change_interval = random.randint(5, 10)
                local_target_offset = random.uniform(-math.pi/2, math.pi/2)
                print(f"    用户 {user_id} 在步 {step} 改变探索方向 (偏移={math.degrees(local_target_offset):.0f}°)")
            
            # === 计算期望方向 ===
            
            # 1. 斜对角线分量（基础方向）
            diagonal_component = diagonal_direction
            
            # 2. 探索分量（局部目标）
            exploration_direction = diagonal_direction + local_target_offset
            exploration_direction = self._normalize_angle(exploration_direction)
            
            # 3. 混合斜对角线和探索方向
            desired_direction = self._blend_angles(
                exploration_direction,
                diagonal_component,
                self.diagonal_bias
            )
            
            # 4. 添加随机漂移
            random_drift = random.uniform(-math.pi / 6, math.pi / 6)
            desired_direction += random_drift * (1 - self.diagonal_bias)
            desired_direction = self._normalize_angle(desired_direction)
            
            # 应用惯性平滑
            desired_direction = self._smooth_angle_transition(
                current_velocity_angle,
                desired_direction,
                self.inertia_strength
            )
            
            # === 边界处理 ===
            margin = 30
            correction_strength = 0.0
            correction_direction = desired_direction
            
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
            
            if correction_strength > 0:
                desired_direction = self._blend_angles(
                    desired_direction,
                    correction_direction,
                    correction_strength
                )
            
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
        """模式2：斜对角线漫游"""
        trajectory = np.zeros((self.num_steps, 3))
        
        x, y = start_x, start_y
        max_distance = self.user_max_speed * self.time_step
        
        diagonal_direction = self._get_skew_diagonal_direction(group)
        current_velocity_angle = random.uniform(0, 2 * math.pi)
        
        wander_angle = 0
        
        for step in range(self.num_steps):
            trajectory[step] = [x, y, random.uniform(self.min_task_size, self.max_task_size)]
            
            wander_angle += random.uniform(-math.pi / 6, math.pi / 6)
            wander_angle = np.clip(wander_angle, -math.pi, math.pi)
            
            if step % 8 == 0 and random.random() < 0.4:
                wander_angle *= 0.5
            
            exploration_direction = diagonal_direction + wander_angle
            exploration_direction = self._normalize_angle(exploration_direction)
            
            desired_direction = self._blend_angles(
                exploration_direction,
                diagonal_direction,
                self.diagonal_bias * 0.7
            )
            
            desired_direction = self._smooth_angle_transition(
                current_velocity_angle,
                desired_direction,
                self.inertia_strength
            )
            
            margin = 30
            if x < margin or x > self.area_length - margin or \
               y < margin or y > self.area_width - margin:
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
        """模式3：斜对角线扫描"""
        trajectory = np.zeros((self.num_steps, 3))
        
        x, y = start_x, start_y
        max_distance = self.user_max_speed * self.time_step
        
        diagonal_direction = self._get_skew_diagonal_direction(group)
        current_velocity_angle = diagonal_direction
        
        sweep_phase = 0
        sweep_frequency = random.uniform(0.1, 0.2)
        sweep_amplitude = random.uniform(0.5, 0.8)
        
        for step in range(self.num_steps):
            trajectory[step] = [x, y, random.uniform(self.min_task_size, self.max_task_size)]
            
            sweep_phase += sweep_frequency
            
            # 横向扫描（垂直于斜对角线）
            perpendicular_angle = diagonal_direction + math.pi / 2
            sweep_offset = math.sin(sweep_phase) * sweep_amplitude * math.pi / 2
            
            forward_component = diagonal_direction
            lateral_component = perpendicular_angle + sweep_offset
            
            desired_direction = self._blend_angles(
                lateral_component,
                forward_component,
                self.diagonal_bias
            )
            
            desired_direction = self._smooth_angle_transition(
                current_velocity_angle,
                desired_direction,
                self.inertia_strength
            )
            
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
        
        # 🆕 绘制对角线参考线（从左上到右下）
        # 主对角线：从(0, area_width)到(area_length, 0)
        ax.plot([0, self.area_length], [self.area_width, 0], 
               'k:', linewidth=2, alpha=0.3, label='主对角线(左上→右下)')
        
        # 标注起始区域
        # 组A起点：左上角
        rect_A = plt.Rectangle(
            (0.05 * self.area_length, 0.70 * self.area_width),
            0.25 * self.area_length, 0.25 * self.area_width,
            fill=True, facecolor='lightcoral', alpha=0.15,
            edgecolor='red', linewidth=2, linestyle='--',
            label='组A起点(左上)'
        )
        ax.add_patch(rect_A)
        
        # 组B起点：右下角
        rect_B = plt.Rectangle(
            (0.70 * self.area_length, 0.05 * self.area_width),
            0.25 * self.area_length, 0.25 * self.area_width,
            fill=True, facecolor='lightblue', alpha=0.15,
            edgecolor='blue', linewidth=2, linestyle='--',
            label='组B起点(右下)'
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
                label = f'用户 {user_id} (A组左上→右下)'
            else:
                color_idx = self.group_division['B'].index(user_id)
                color = colors_B[color_idx]
                label = f'用户 {user_id} (B组右下→左上)'
            
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
        ax.set_title(f'对角线探索轨迹 (左上→右下, 偏好={self.diagonal_bias})\n'
                    f'{self.num_users}个用户 × {self.num_steps}步 | 模式: {self.movement_mode}', 
                    fontsize=20, fontweight='bold', pad=25)
        
        # 图例
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
    print("对角线探索轨迹生成器 (左上→右下)")
    print("="*70)
    
    generator = LeftTopToRightBottomTrajectoryGenerator(
        num_users=5,
        num_steps=31,
        area_length=400,
        area_width=400,
        user_max_speed=2.0,
        time_step=5.0,
        min_task_size=5,
        max_task_size=10,
        movement_mode='diagonal_explore',  # 推荐
        # movement_mode='diagonal_wander',
        # movement_mode='diagonal_sweep',
        diagonal_bias=0.1,  # 对角线偏好
        exploration_range=0.95,  # 探索范围
        inertia_strength=0.7,
    )
    
    print("\n正在生成对角线探索轨迹...")
    trajectories = generator.generate_trajectories()
    
    # 验证轨迹特性
    print("\n验证轨迹特性:")
    for user_id, traj in trajectories.items():
        start_x, start_y = traj[0, 0], traj[0, 1]
        end_x, end_y = traj[-1, 0], traj[-1, 1]
        
        dx = end_x - start_x
        dy = end_y - start_y
        
        if user_id in generator.group_division['A']:
            group = 'A (左上→右下↘)'
            expected_dx = "正(向右)"
            expected_dy = "负(向下)"
        else:
            group = 'B (右下→左上↖)'
            expected_dx = "负(向左)"
            expected_dy = "正(向上)"
        
        print(f"  用户 {user_id} ({group}): "
              f"位移=({dx:+.0f}, {dy:+.0f}), "
              f"期望方向=({expected_dx}, {expected_dy})")
    
    generator.save_trajectories(trajectories, "user_trajectories_hot.json")
    generator.plot_trajectories(trajectories, "user_trajectories_hot.png")
    
    print(f"\n✅ 完成！用户从左上角到右下角对角线移动，在周围区域自由探索！")