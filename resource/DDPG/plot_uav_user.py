"""
无人机和用户轨迹可视化工具
同时显示UAV和用户的移动轨迹
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os

class UAVUserTrajectoryPlotter:
    """无人机和用户轨迹联合绘图器"""
    
    def __init__(self, area_length=200, area_width=200):
        """
        初始化绘图器
        
        Args:
            area_length: 区域长度（米）
            area_width: 区域宽度（米）
        """
        self.area_length = area_length
        self.area_width = area_width
    
    def load_uav_trajectories(self, json_file):
        """
        从JSON文件加载无人机真实轨迹
        
        Args:
            json_file: 无人机动作日志JSON文件路径
            
        Returns:
            dict: {uav_id: trajectory_array [N_steps, 2]}
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_uavs = len(data['initial_positions'])
        trajectories = {}
        
        for uav_id in range(num_uavs):
            uav_key = f'uav_{uav_id}'
            
            # 完整轨迹 = 初始位置 + 每步移动后的位置
            trajectory = [data['initial_positions'][uav_key]]
            
            for step_data in data['actions']:
                trajectory.append(step_data['actions'][uav_key]['position'])
            
            trajectories[uav_id] = np.array(trajectory)
        
        print(f"✓ 加载 {num_uavs} 架无人机轨迹，每架 {len(trajectories[0])} 个点")
        
        return trajectories, data
    
    def load_user_trajectories(self, json_file):
        """
        从JSON文件加载用户轨迹
        
        Args:
            json_file: 用户轨迹JSON文件路径
            
        Returns:
            dict: {user_id: trajectory_array [N_steps, 3]}  # [x, y, task_size]
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trajectories = {}
        for user_id_str, traj_list in data['trajectories'].items():
            user_id = int(user_id_str)
            trajectories[user_id] = np.array(traj_list)
        
        print(f"✓ 加载 {len(trajectories)} 个用户轨迹，每个 {len(trajectories[0])} 个点")
        
        return trajectories, data
    
    def plot_combined_trajectories(self, uav_json_file, user_json_file, 
                                   save_path=None, episode=None):
        """
        绘制无人机和用户的联合轨迹图
        
        Args:
            uav_json_file: 无人机轨迹JSON文件
            user_json_file: 用户轨迹JSON文件
            save_path: 保存路径
            episode: episode编号
        """
        # 加载数据
        uav_trajectories, uav_data = self.load_uav_trajectories(uav_json_file)
        user_trajectories, user_data = self.load_user_trajectories(user_json_file)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # === 1. 绘制用户轨迹（底层，浅色虚线）===
        user_colors = plt.cm.Pastel1(np.linspace(0, 1, len(user_trajectories)))
        
        for user_id, user_traj in user_trajectories.items():
            color = user_colors[user_id]
            
            # 用户轨迹用细虚线
            ax.plot(user_traj[:, 0], user_traj[:, 1], 
                   color=color, linewidth=1.5, linestyle='--', alpha=0.5,
                   label=f'用户 {user_id}')
            
            # 用户起点（小圆点）
            ax.plot(user_traj[0, 0], user_traj[0, 1], 
                   'o', color=color, markersize=8, alpha=0.7)
            
            # 用户终点（小方块）
            ax.plot(user_traj[-1, 0], user_traj[-1, 1], 
                   's', color=color, markersize=8, alpha=0.7)
        
        # === 2. 绘制无人机轨迹（上层，深色实线）===
        uav_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for uav_id, uav_traj in uav_trajectories.items():
            color = uav_colors[uav_id % len(uav_colors)]
            
            # 无人机轨迹用粗实线
            ax.plot(uav_traj[:, 0], uav_traj[:, 1], 
                   color=color, linewidth=3.5, alpha=0.9,
                   label=f'UAV {uav_id}', zorder=10)
            
            # 无人机起点（大圆圈，绿色边框）
            ax.plot(uav_traj[0, 0], uav_traj[0, 1], 
                   'o', color=color, markersize=22, 
                   markeredgecolor='green', markeredgewidth=4,
                   zorder=15)
            
            # 无人机终点（大方块，红色边框）
            ax.plot(uav_traj[-1, 0], uav_traj[-1, 1], 
                   's', color=color, markersize=22,
                   markeredgecolor='red', markeredgewidth=4,
                   zorder=15)
            
            # 标注起点和终点
            ax.text(uav_traj[0, 0], uav_traj[0, 1] + 6, 
                   f'UAV{uav_id}起点', fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                           edgecolor=color, linewidth=2, alpha=0.9),
                   zorder=20)
            
            ax.text(uav_traj[-1, 0], uav_traj[-1, 1] - 10, 
                   f'UAV{uav_id}终点', fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                           edgecolor=color, linewidth=2, alpha=0.9),
                   zorder=20)
        
        # === 3. 绘制区域边界 ===
        boundary = patches.Rectangle((0, 0), self.area_length, self.area_width,
                                     linewidth=3, edgecolor='black', 
                                     facecolor='none', linestyle='--',
                                     label='区域边界')
        ax.add_patch(boundary)
        
        # === 4. 设置坐标轴和标题 ===
        ax.set_xlim(-15, self.area_length + 15)
        ax.set_ylim(-15, self.area_width + 15)
        ax.set_xlabel('X 坐标 (米)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y 坐标 (米)', fontsize=14, fontweight='bold')
        
        # 标题
        title = f'无人机与用户移动轨迹'
        if episode is not None:
            title += f' (Episode {episode})'
        ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
        
        # === 5. 图例（分两列显示）===
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95, ncol=2)
        
        # === 6. 网格 ===
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # === 7. 等比例坐标轴 ===
        ax.set_aspect('equal')
        
        # === 8. 添加统计信息文本框 ===
        stats_text = self._generate_stats_text(uav_trajectories, user_trajectories)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # === 9. 紧凑布局 ===
        plt.tight_layout()
        
        # === 10. 保存图片 ===
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            episode_str = f'_episode_{episode}' if episode is not None else ''
            save_path = f'uav_user_trajectories{episode_str}_{timestamp}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 轨迹图已保存: {save_path}")
        
        return save_path
    
    def plot_trajectory_with_time_snapshots(self, uav_json_file, user_json_file,
                                           save_path=None, episode=None, 
                                           num_snapshots=5):
        """
        绘制带时间快照的轨迹图（显示不同时刻的位置关系）
        
        Args:
            uav_json_file: 无人机轨迹JSON文件
            user_json_file: 用户轨迹JSON文件
            save_path: 保存路径
            episode: episode编号
            num_snapshots: 快照数量
        """
        # 加载数据
        uav_trajectories, uav_data = self.load_uav_trajectories(uav_json_file)
        user_trajectories, user_data = self.load_user_trajectories(user_json_file)
        
        # 创建子图
        fig = plt.figure(figsize=(20, 12))
        
        # 计算快照时刻
        num_steps = len(next(iter(uav_trajectories.values())))
        snapshot_steps = np.linspace(0, num_steps - 1, num_snapshots, dtype=int)
        
        for idx, step in enumerate(snapshot_steps):
            ax = plt.subplot(2, 3, idx + 1)
            
            # 绘制用户位置（当前时刻）
            user_colors = plt.cm.Pastel1(np.linspace(0, 1, len(user_trajectories)))
            for user_id, user_traj in user_trajectories.items():
                if step < len(user_traj):
                    color = user_colors[user_id]
                    ax.plot(user_traj[step, 0], user_traj[step, 1], 
                           'o', color=color, markersize=12, alpha=0.7,
                           label=f'用户{user_id}')
            
            # 绘制无人机位置（当前时刻）
            uav_colors = ['#e74c3c', '#3498db', '#2ecc71']
            for uav_id, uav_traj in uav_trajectories.items():
                if step < len(uav_traj):
                    color = uav_colors[uav_id % len(uav_colors)]
                    ax.plot(uav_traj[step, 0], uav_traj[step, 1], 
                           '^', color=color, markersize=18, 
                           markeredgecolor='black', markeredgewidth=2,
                           label=f'UAV{uav_id}', zorder=10)
                    
                    # 绘制覆盖范围圈（可选）
                    circle = plt.Circle((uav_traj[step, 0], uav_traj[step, 1]), 
                                       30, color=color, alpha=0.1, zorder=5)
                    ax.add_patch(circle)
            
            # 设置
            ax.set_xlim(0, self.area_length)
            ax.set_ylim(0, self.area_width)
            ax.set_xlabel('X (米)', fontsize=10)
            ax.set_ylabel('Y (米)', fontsize=10)
            ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # 总标题
        title = f'无人机与用户位置快照'
        if episode is not None:
            title += f' (Episode {episode})'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            episode_str = f'_episode_{episode}' if episode is not None else ''
            save_path = f'snapshots{episode_str}_{timestamp}.png'
        
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ 快照图已保存: {save_path}")
        
        return save_path
    
    def _generate_stats_text(self, uav_trajectories, user_trajectories):
        """生成统计信息文本"""
        stats = []
        stats.append(f"无人机数量: {len(uav_trajectories)}")
        stats.append(f"用户数量: {len(user_trajectories)}")
        
        # 计算UAV总移动距离
        for uav_id, traj in uav_trajectories.items():
            distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
            total_dist = np.sum(distances)
            stats.append(f"UAV{uav_id}总距离: {total_dist:.1f}m")
        
        return '\n'.join(stats)


# ============================================================================
# 使用示例
# ============================================================================

def plot_single_episode(uav_json, user_json, output_dir='trajectory_plots'):
    """绘制单个episode的轨迹"""
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = UAVUserTrajectoryPlotter(area_length=400, area_width=400)
    
    # 从文件名提取episode编号
    import re
    match = re.search(r'episode_(\d+)', uav_json)
    episode = int(match.group(1)) if match else None
    
    # 1. 绘制完整轨迹图
    save_path1 = os.path.join(output_dir, f'episode_{episode}_full_trajectory.png')
    plotter.plot_combined_trajectories(uav_json, user_json, 
                                       save_path=save_path1, episode=episode)
    
    # 2. 绘制时间快照图
    save_path2 = os.path.join(output_dir, f'episode_{episode}_snapshots.png')
    plotter.plot_trajectory_with_time_snapshots(uav_json, user_json,
                                               save_path=save_path2, episode=episode,
                                               num_snapshots=6)
    
    print(f"\n✓ Episode {episode} 可视化完成")


def plot_all_episodes(action_logs_dir='action_logs', 
                     user_json='/home/niuma008/zsz/MultiUav_1025/user_trajectories.json',
                     output_dir='trajectory_plots'):
    """批量绘制所有episode"""
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = UAVUserTrajectoryPlotter(area_length=400, area_width=400)
    
    # 查找所有episode的JSON文件
    json_files = [f for f in os.listdir(action_logs_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"\n找到 {len(json_files)} 个episode")
    print("=" * 60)
    
    for json_file in json_files:
        uav_json_path = os.path.join(action_logs_dir, json_file)
        
        # 提取episode编号
        import re
        match = re.search(r'episode_(\d+)', json_file)
        episode = int(match.group(1)) if match else None
        
        print(f"\n处理 Episode {episode}...")
        
        # 绘制完整轨迹图
        save_path = os.path.join(output_dir, f'episode_{episode}_combined.png')
        plotter.plot_combined_trajectories(uav_json_path, user_json,
                                          save_path=save_path, episode=episode)
    
    print(f"\n✓ 所有轨迹图已保存到: {output_dir}/")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("无人机与用户轨迹联合可视化工具")
    print("=" * 60)
    
    # 用户轨迹文件路径
    USER_TRAJ_FILE = '/home/niuma008/zsz/MultiUav_1025/user_trajectories.json'
    
    # 方式1: 绘制单个episode
    if len(sys.argv) > 1:
        uav_json_file = sys.argv[1]
        print(f"\n单文件模式: {uav_json_file}")
        plot_single_episode(uav_json_file, USER_TRAJ_FILE)
    
    # 方式2: 批量处理所有episode
    else:
        print("\n批量处理模式")
        plot_all_episodes(
            action_logs_dir='action_logs',
            user_json=USER_TRAJ_FILE,
            output_dir='trajectory_plots'
        )
    
    print("\n完成！")