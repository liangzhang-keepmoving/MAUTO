"""
用户分配模块 - 处理UAV与用户的竞争分配机制
"""
import numpy as np

class UserAllocationManager:
    """用户分配管理器 - 处理UAV间的用户竞争和分配"""
    
    def __init__(self, num_uavs, num_users):
        """
        初始化用户分配管理器
        
        Args:
            num_uavs: UAV数量
            num_users: 用户数量
        """
        self.num_uavs = num_uavs
        self.num_users = num_users
        
    def process_uav_actions_with_conflict_resolution(self, actions):
        """
        从actions中提取用户分配结果
        
        Args:
            actions: 动作字典，格式为 {f'uav_{uav_id}': {'user_competition_probs': array, ...}}
            num_users: 用户数量
            num_uavs: 无人机数量
            
        Returns:
            dict: 用户分配结果 {user_id: uav_id}
        """
        if actions is None:
            return {}
        
        # 构建竞争概率矩阵 [num_uavs, num_users]
        competition_matrix = np.zeros((self.num_uavs, self.num_users))
        
        for uav_id in range(self.num_uavs):
            uav_key = f'uav_{uav_id}'
            if uav_key in actions and 'user_competition_probs' in actions[uav_key]:
                probs = actions[uav_key]['user_competition_probs']
                # 确保概率数组长度正确
                if len(probs) == self.num_users:
                    competition_matrix[uav_id, :] = probs
                else:
                    # 如果长度不匹配，截取或填充
                    min_len = min(len(probs), self.num_users)
                    competition_matrix[uav_id, :min_len] = probs[:min_len]
        
        # 为每个用户选择竞争概率最高的无人机
        user_assignments = {}
        for user_id in range(self.num_users):
            # 获取所有无人机对该用户的竞争概率
            user_probs = competition_matrix[:, user_id]
            
            # 选择概率最高的无人机
            if np.sum(user_probs) > 0:  # 确保至少有一个无人机有正概率
                assigned_uav = np.argmax(user_probs)
                user_assignments[user_id] = assigned_uav
            else:
                # 如果所有概率都是0，随机分配给第一个无人机
                user_assignments[user_id] = 0
        
        return user_assignments

