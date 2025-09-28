"""
用户分配模块 - 处理UAV与用户的竞争分配机制
"""
import numpy as np

class UserAllocationManager:
    """用户分配管理器 - 处理UAV间的用户竞争和分配"""
    
    def __init__(self, num_uavs, num_users, max_task_size=10.0):
        """
        初始化用户分配管理器
        
        Args:
            num_uavs: UAV数量
            num_users: 用户数量  
            max_task_size: 最大任务大小 (MB)
        """
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.max_task_size = max_task_size
        
    def process_uav_actions_with_conflict_resolution(self, actions, user_states):
        """
        处理UAV动作并解决用户分配冲突
        
        Args:
            actions: 动作字典，包含每个UAV的用户竞争概率等
            user_states: 用户状态数组 [x, y, task_size]
            
        Returns:
            tuple: (user_allocation_rewards, user_assignments)
                - user_allocation_rewards: 用户分配奖励字典 {f'uav_{uav_id}': reward}
                - user_assignments: 用户分配结果 {user_id: uav_id}
        """
        if actions is None:
            return {f'uav_{i}': 0.0 for i in range(self.num_uavs)}, {}
        
        user_allocation_rewards = {f'uav_{i}': 0.0 for i in range(self.num_uavs)}
        
        # 1. 收集所有UAV对每个用户的竞争概率
        prob_matrix = np.zeros((self.num_uavs, self.num_users))
        
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            if uav_key in actions and 'user_competition_probs' in actions[uav_key]:
                competition_probs = np.array(actions[uav_key]['user_competition_probs'])
                prob_matrix[uav_id, :] = competition_probs
            else:
                # 兼容旧格式：如果有user_selection，转换为概率
                if uav_key in actions and 'user_selection' in actions[uav_key]:
                    user_selection = np.array(actions[uav_key]['user_selection'])
                    prob_matrix[uav_id, :] = user_selection.astype(float)
        
        # 2. 竞争分配：每个用户分配给概率最高的UAV
        final_assignments = {}
        
        for user_id in range(self.num_users):
            # 获取所有UAV对该用户的竞争概率
            user_probs = prob_matrix[:, user_id]
            
            # 找到概率最高的UAV
            winner_uav = np.argmax(user_probs)
            winning_prob = user_probs[winner_uav]
            
            # 分配给概率最高的UAV
            final_assignments[user_id] = winner_uav
            
            # 3. 计算奖励：只给获胜UAV奖励，失败UAV不奖励也不惩罚
            task_size = user_states[user_id, 2]
            
            # 获胜奖励：只有分配到用户的UAV获得奖励
            base_reward = 1.0
            size_bonus = task_size / self.max_task_size * 0.5
            total_winner_reward = base_reward + size_bonus
            user_allocation_rewards[f'uav_{winner_uav}'] += total_winner_reward
            
            # 失败UAV不给任何奖励或惩罚，通过"得不到奖励"来体现竞争压力
        
        return user_allocation_rewards, final_assignments
