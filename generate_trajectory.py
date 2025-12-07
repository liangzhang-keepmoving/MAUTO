"""
改进的用户轨迹生成器 - 简化版（仅保存数据和图片）
"""
import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
from typing import Dict


class ImprovedUserTrajectoryGenerator:
    """改进的用户轨迹生成器 - 支持多种移动模式"""
    
    def __init__(
        self, 
        num_users: int = 5,
        num_steps: int = 30,
        area_length: float = 150.0,
        area_width: float = 150.0,
        user_max_speed: float = 2.0,
        time_step: float = 5.0,
        min_task_size: float = 0.5,
        max_task_size: float = 1.0,
        movement_mode: str = 'inertia',  # 'random', 'inertia', 'target'
        seed: int = None
    ):
        """
        初始化轨迹生成器
        
        Args:
            movement_mode: 移动模式
                - 'random': 纯随机方向
                - 'inertia': 带惯性的移动（推荐）
                - 'target': 目标导向移动
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
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _generate_dispersed_starting_points(self) -> list:
        """
        生成均匀分散的起点位置
        使用网格+随机偏移的方法确保起点分散
        """
        starting_points = []
        
        # 计算网格尺寸
        grid_cols = int(np.ceil(np.sqrt(self.num_users)))
        grid_rows = int(np.ceil(self.num_users / grid_cols))
        
        # 计算每个网格的大小
        cell_width = self.area_length / grid_cols
        cell_height = self.area_width / grid_rows
        
        # 在每个网格中随机放置一个起点
        user_count = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if user_count >= self.num_users:
                    break
                
                # 网格中心位置
                center_x = (col + 0.5) * cell_width
                center_y = (row + 0.5) * cell_height
                
                # 在网格内添加随机偏移（偏移范围为网格大小的30%）
                offset_x = random.uniform(-0.3 * cell_width, 0.3 * cell_width)
                offset_y = random.uniform(-0.3 * cell_height, 0.3 * cell_height)
                
                x = max(10, min(self.area_length - 10, center_x + offset_x))
                y = max(10, min(self.area_width - 10, center_y + offset_y))
                
                starting_points.append((x, y))
                user_count += 1
            
            if user_count >= self.num_users:
                break
        
        return starting_points
    
    def generate_trajectories(self) -> Dict[int, np.ndarray]:
        """生成所有用户的轨迹"""
        trajectories = {}
        
        # 生成均匀分散的起点
        starting_points = self._generate_dispersed_starting_points()
        
        for user_id in range(self.num_users):
            start_x, start_y = starting_points[user_id]
            
            if self.movement_mode == 'random':
                trajectory = self._generate_random_walk(user_id, start_x, start_y)
            elif self.movement_mode == 'inertia':
                trajectory = self._generate_inertia_walk(user_id, start_x, start_y)
            elif self.movement_mode == 'target':
                trajectory = self._generate_target_walk(user_id, start_x, start_y)
            else:
                raise ValueError(f"Unknown movement_mode: {self.movement_mode}")
            
            trajectories[user_id] = trajectory
        
        return trajectories
    
    def _generate_random_walk(self, user_id: int, start_x: float, start_y: float) -> np.ndarray:
        """模式1：纯随机方向（修正版）"""
        trajectory = np.zeros((self.num_steps, 3))
        
        x = start_x
        y = start_y
        
        max_distance = self.user_max_speed * self.time_step
        
        for step in range(self.num_steps):
            trajectory[step, 0] = x
            trajectory[step, 1] = y
            trajectory[step, 2] = random.uniform(self.min_task_size, self.max_task_size)
            
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.7 * max_distance, max_distance)
            
            new_x = x + distance * math.cos(angle)
            new_y = y + distance * math.sin(angle)
            
            if new_x < 0:
                new_x = -new_x
                angle = math.pi - angle
            elif new_x > self.area_length:
                new_x = 2 * self.area_length - new_x
                angle = math.pi - angle
            
            if new_y < 0:
                new_y = -new_y
                angle = -angle
            elif new_y > self.area_width:
                new_y = 2 * self.area_width - new_y
                angle = -angle
            
            x = max(0, min(self.area_length, new_x))
            y = max(0, min(self.area_width, new_y))
        
        return trajectory
    
    def _generate_inertia_walk(self, user_id: int, start_x: float, start_y: float) -> np.ndarray:
        """模式2：带惯性的随机游走（推荐）"""
        trajectory = np.zeros((self.num_steps, 3))
        
        x = start_x
        y = start_y
        
        angle = random.uniform(0, 2 * math.pi)
        max_distance = self.user_max_speed * self.time_step
        
        for step in range(self.num_steps):
            trajectory[step, 0] = x
            trajectory[step, 1] = y
            trajectory[step, 2] = random.uniform(self.min_task_size, self.max_task_size)
            
            # 保持大致方向，加入 ±45° 扰动
            angle_change = random.uniform(-math.pi / 4, math.pi / 4)
            angle += angle_change
            
            distance = random.uniform(0.8 * max_distance, max_distance)
            
            new_x = x + distance * math.cos(angle)
            new_y = y + distance * math.sin(angle)
            
            if new_x < 0 or new_x > self.area_length:
                angle = math.pi - angle
                new_x = max(0, min(self.area_length, new_x))
            
            if new_y < 0 or new_y > self.area_width:
                angle = -angle
                new_y = max(0, min(self.area_width, new_y))
            
            x = new_x
            y = new_y
        
        return trajectory
    
    def _generate_target_walk(self, user_id: int, start_x: float, start_y: float) -> np.ndarray:
        """模式3：目标导向移动"""
        trajectory = np.zeros((self.num_steps, 3))
        
        x = start_x
        y = start_y
        
        target_x = random.uniform(0, self.area_length)
        target_y = random.uniform(0, self.area_width)
        
        max_distance = self.user_max_speed * self.time_step
        target_reached_threshold = max_distance * 1.5
        
        for step in range(self.num_steps):
            trajectory[step, 0] = x
            trajectory[step, 1] = y
            trajectory[step, 2] = random.uniform(self.min_task_size, self.max_task_size)
            
            dx = target_x - x
            dy = target_y - y
            distance_to_target = math.sqrt(dx**2 + dy**2)
            
            if distance_to_target < target_reached_threshold or step % 5 == 0:
                target_x = random.uniform(0, self.area_length)
                target_y = random.uniform(0, self.area_width)
                dx = target_x - x
                dy = target_y - y
                distance_to_target = math.sqrt(dx**2 + dy**2)
            
            if distance_to_target > 0:
                angle = math.atan2(dy, dx)
                angle += random.uniform(-math.pi / 12, math.pi / 12)
            else:
                angle = random.uniform(0, 2 * math.pi)
            
            distance = random.uniform(0.9 * max_distance, max_distance)
            distance = min(distance, distance_to_target)
            
            new_x = x + distance * math.cos(angle)
            new_y = y + distance * math.sin(angle)
            
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
                'movement_mode': self.movement_mode
            },
            'trajectories': {}
        }
        
        for user_id, traj in trajectories.items():
            json_data['trajectories'][str(user_id)] = traj.tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 轨迹数据已保存: {filepath}")
    
    def plot_trajectories(self, trajectories: Dict, save_path: str = "user_trajectories.png"):
        """绘制并保存轨迹图片（不显示）"""
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_users))
        
        for user_id, traj in trajectories.items():
            x_coords = traj[:, 0]
            y_coords = traj[:, 1]
            
            plt.plot(x_coords, y_coords, '-', color=colors[user_id], 
                    linewidth=2.5, alpha=0.7, label=f'用户 {user_id}')
            
            plt.plot(x_coords[0], y_coords[0], 'o', color=colors[user_id], 
                    markersize=15, markeredgecolor='green', markeredgewidth=4)
            
            plt.plot(x_coords[-1], y_coords[-1], 's', color=colors[user_id], 
                    markersize=15, markeredgecolor='red', markeredgewidth=4)
        
        # 修改：显示完整的坐标轴范围 0 到 area_length/area_width
        plt.xlim(0, self.area_length)
        plt.ylim(0, self.area_width)
        
        plt.xlabel('X坐标 (米)', fontsize=14)
        plt.ylabel('Y坐标 (米)', fontsize=14)
        plt.title(f'{self.num_users}个用户的{self.num_steps}步轨迹 (模式: {self.movement_mode})', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        print(f"✓ 轨迹图片已保存: {save_path}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建生成器（使用推荐配置）
    generator = ImprovedUserTrajectoryGenerator(
        num_users=5,
        num_steps=16,
        area_length=400,
        area_width=400,
        user_max_speed=2.0,
        time_step=5.0,
        min_task_size=15,
        max_task_size=20,
        movement_mode='random',  # 推荐使用 'inertia'
    )
    
    # 生成轨迹
    print("正在生成轨迹...")
    trajectories = generator.generate_trajectories()
    print(f"✓ 已生成 {len(trajectories)} 个用户的轨迹")
    
    # 保存数据和图片
    generator.save_trajectories(trajectories, "user_trajectories.json")
    generator.plot_trajectories(trajectories, "user_trajectories.png")
    
    print("\n完成！")