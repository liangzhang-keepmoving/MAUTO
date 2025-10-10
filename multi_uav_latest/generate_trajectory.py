import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
from typing import Dict


class UserTrajectoryGenerator:
    """用户轨迹生成器 - 预生成一个episode的所有用户轨迹（随机游走）"""
    
    def __init__(
        self, 
        num_users: int = 5,
        num_steps: int = 30,
        area_length: float = 500.0,
        area_width: float = 500.0,
        user_max_speed: float = 2.0,
        time_step: float = 5.0,
        min_task_size: float = 0.5,
        max_task_size: float = 1.0,
        seed: int = None
    ):
        """
        初始化轨迹生成器
        
        Args:
            num_users: 用户数量
            num_steps: 轨迹步数
            area_length: 区域长度(米)
            area_width: 区域宽度(米)
            user_max_speed: 用户最大移动速度(m/s)
            time_step: 时间步长(秒)
            min_task_size: 最小任务大小(MB)
            max_task_size: 最大任务大小(MB)
            seed: 随机种子(可选)
        """
        self.num_users = num_users
        self.num_steps = num_steps
        self.area_length = area_length
        self.area_width = area_width
        self.user_max_speed = user_max_speed
        self.time_step = time_step
        self.min_task_size = min_task_size
        self.max_task_size = max_task_size
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_trajectories(self) -> Dict[int, np.ndarray]:
        """
        生成所有用户的随机游走轨迹
        
        Returns:
            trajectories: dict {user_id: trajectory_array}
                trajectory_array shape: (num_steps, 3) - [x, y, task_size]
        """
        trajectories = {}
        
        for user_id in range(self.num_users):
            trajectory = np.zeros((self.num_steps, 3))
            
            # 随机初始位置
            x = random.uniform(0, self.area_length)
            y = random.uniform(0, self.area_width)
            
            # 随机游走
            for step in range(self.num_steps):
                # 记录当前位置和任务
                trajectory[step, 0] = x
                trajectory[step, 1] = y
                trajectory[step, 2] = random.uniform(self.min_task_size, self.max_task_size)
                
                # 随机移动
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(8, 12)
                
                # 计算新位置
                new_x = x + distance * math.cos(angle)
                new_y = y + distance * math.sin(angle)
                
                # 边界限制
                x = max(0, min(self.area_length, new_x))
                y = max(0, min(self.area_width, new_y))
            
            trajectories[user_id] = trajectory
        
        return trajectories
    
    def save_trajectories(self, trajectories: Dict, filepath: str = "user_trajectories.json"):
        """
        保存轨迹到JSON文件
        
        Args:
            trajectories: 轨迹字典
            filepath: 保存路径
        """
        # 转换numpy数组为列表（JSON可序列化）
        json_data = {
            'metadata': {
                'num_users': self.num_users,
                'num_steps': self.num_steps,
                'area_length': self.area_length,
                'area_width': self.area_width,
                'user_max_speed': self.user_max_speed,
                'time_step': self.time_step,
                'min_task_size': self.min_task_size,
                'max_task_size': self.max_task_size
            },
            'trajectories': {}
        }
        
        for user_id, traj in trajectories.items():
            json_data['trajectories'][str(user_id)] = traj.tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 轨迹已保存到: {filepath}")
    
    @staticmethod
    def load_trajectories(filepath: str = "user_trajectories.json") -> Dict:
        """
        从JSON文件加载轨迹
        
        Args:
            filepath: 文件路径
        
        Returns:
            trajectories: 轨迹字典 {user_id: numpy.ndarray}
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        trajectories = {}
        for user_id_str, traj_list in json_data['trajectories'].items():
            user_id = int(user_id_str)
            trajectories[user_id] = np.array(traj_list)
        
        print(f"✓ 轨迹已从 {filepath} 加载")
        return trajectories
    
    def plot_trajectories(self, trajectories: Dict, save_path: str = "user_trajectories.png"):
        """
        绘制并保存用户轨迹图
        
        Args:
            trajectories: 轨迹字典
            save_path: 图片保存路径
        """
        plt.figure(figsize=(12, 10))
        
        # 颜色列表
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_users))
        
        for user_id, traj in trajectories.items():
            x_coords = traj[:, 0]
            y_coords = traj[:, 1]
            
            # 绘制轨迹线
            plt.plot(x_coords, y_coords, '-', color=colors[user_id], 
                    linewidth=2, alpha=0.7, label=f'用户 {user_id}')
            
            # 标记起点（大绿圈）
            plt.plot(x_coords[0], y_coords[0], 'o', color=colors[user_id], 
                    markersize=12, markeredgecolor='green', markeredgewidth=3)
            
            # 标记终点（大红方块）
            plt.plot(x_coords[-1], y_coords[-1], 's', color=colors[user_id], 
                    markersize=12, markeredgecolor='red', markeredgewidth=3)
            
            # 添加起点和终点文字标注
            plt.text(x_coords[0], y_coords[0], f'  起{user_id}', 
                    fontsize=10, color='green', fontweight='bold')
            plt.text(x_coords[-1], y_coords[-1], f'  终{user_id}', 
                    fontsize=10, color='red', fontweight='bold')
        
        # 设置图形属性
        plt.xlim(-10, self.area_length + 10)
        plt.ylim(-10, self.area_width + 10)
        plt.xlabel('X坐标 (米)', fontsize=14)
        plt.ylabel('Y坐标 (米)', fontsize=14)
        plt.title(f'{self.num_users}个用户的{self.num_steps}步随机游走轨迹', fontsize=16, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axis('equal')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 轨迹图已保存到: {save_path}")
        
        plt.close()
    
    def print_statistics(self, trajectories: Dict):
        """
        打印轨迹统计信息
        
        Args:
            trajectories: 轨迹字典
        """
        print("\n" + "="*70)
        print("轨迹统计信息")
        print("="*70)
        
        for user_id, traj in trajectories.items():
            # 计算总移动距离
            total_distance = 0
            for step in range(1, self.num_steps):
                dx = traj[step, 0] - traj[step-1, 0]
                dy = traj[step, 1] - traj[step-1, 1]
                total_distance += math.sqrt(dx**2 + dy**2)
            
            # 任务统计
            tasks = traj[:, 2]
            avg_task = np.mean(tasks)
            total_task = np.sum(tasks)
            
            print(f"\n用户 {user_id}:")
            print(f"  起点: ({traj[0, 0]:.2f}, {traj[0, 1]:.2f})")
            print(f"  终点: ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})")
            print(f"  总移动距离: {total_distance:.2f} 米")
            print(f"  平均任务大小: {avg_task:.3f} MB")
            print(f"  总任务量: {total_task:.2f} MB")
        
        print("\n" + "="*70)


# ================== 使用示例 ==================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("用户轨迹生成器")
    print("="*70)
    
    # 创建生成器
    generator = UserTrajectoryGenerator(
        num_users=5,
        num_steps=25,
        area_length=150,
        area_width=150,
        user_max_speed=1.0,
        time_step=8.0,
        min_task_size=0.5,
        max_task_size=1.0
    )
    
    # 生成轨迹
    print("\n正在生成轨迹...")
    trajectories = generator.generate_trajectories()
    print(f"✓ 已生成 {len(trajectories)} 个用户的轨迹")
    
    # 打印统计信息
    generator.print_statistics(trajectories)
    
    # 保存为JSON
    generator.save_trajectories(trajectories, "user_trajectories.json")
    
    # 绘制并保存轨迹图
    generator.plot_trajectories(trajectories, "user_trajectories.png")
    
    # 测试加载
    print("\n测试加载轨迹...")
    loaded_trajectories = UserTrajectoryGenerator.load_trajectories("user_trajectories.json")
    print(f"✓ 验证数据一致性: {np.allclose(trajectories[0], loaded_trajectories[0])}")
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)