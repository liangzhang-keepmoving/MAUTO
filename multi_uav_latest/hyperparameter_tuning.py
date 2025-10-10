#!/usr/bin/env python3
"""
多无人机DDPG超参数调优脚本
使用网格搜索或随机搜索来找到最佳超参数组合
"""

import os
import json
import itertools
import numpy as np
from datetime import datetime
import torch

from train_ddpg import train_ddpg
from test_and_visualize import test_trained_model


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, base_config, param_ranges, search_type='grid'):
        """
        Args:
            base_config: 基础配置
            param_ranges: 参数搜索范围
            search_type: 搜索类型 ('grid' 或 'random')
        """
        self.base_config = base_config
        self.param_ranges = param_ranges
        self.search_type = search_type
        self.results = []
        
        # 创建结果保存目录
        self.results_dir = f'./hyperparameter_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def generate_configs(self, max_trials=None):
        """生成配置组合"""
        if self.search_type == 'grid':
            return self._grid_search_configs()
        elif self.search_type == 'random':
            return self._random_search_configs(max_trials or 20)
        else:
            raise ValueError(f"不支持的搜索类型: {self.search_type}")
    
    def _grid_search_configs(self):
        """网格搜索配置生成"""
        param_names = list(self.param_ranges.keys())
        param_values = [self.param_ranges[name] for name in param_names]
        
        configs = []
        for combination in itertools.product(*param_values):
            config = self.base_config.copy()
            for i, param_name in enumerate(param_names):
                config[param_name] = combination[i]
            configs.append(config)
        
        return configs
    
    def _random_search_configs(self, num_trials):
        """随机搜索配置生成"""
        configs = []
        
        for _ in range(num_trials):
            config = self.base_config.copy()
            
            for param_name, param_range in self.param_ranges.items():
                if isinstance(param_range[0], float):
                    # 浮点数参数：对数均匀采样
                    if param_range[0] > 0:
                        log_range = [np.log10(param_range[0]), np.log10(param_range[1])]
                        value = 10 ** np.random.uniform(log_range[0], log_range[1])
                    else:
                        value = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range[0], int):
                    # 整数参数
                    value = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    # 离散选择
                    value = np.random.choice(param_range)
                
                config[param_name] = value
            
            configs.append(config)
        
        return configs
    
    def run_tuning(self, max_trials=None):
        """运行超参数调优"""
        configs = self.generate_configs(max_trials)
        
        print(f"=== 开始超参数调优 ===")
        print(f"搜索类型: {self.search_type}")
        print(f"总配置数: {len(configs)}")
        print(f"结果保存目录: {self.results_dir}")
        
        best_score = float('-inf')
        best_config = None
        
        for i, config in enumerate(configs):
            print(f"\n--- 试验 {i+1}/{len(configs)} ---")
            print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
            
            try:
                # 训练模型
                config['log_dir'] = os.path.join(self.results_dir, f'trial_{i+1}')
                config['max_episodes'] = min(config.get('max_episodes', 1000), 500)  # 限制训练轮数
                
                agent, monitor, train_results = train_ddpg(config)
                
                # 评估模型
                model_path = os.path.join(config['log_dir'], 'best_model.pth')
                if os.path.exists(model_path):
                    test_results = test_trained_model(
                        model_path, config, visualize=False, num_episodes=5
                    )
                    
                    # 计算评估分数
                    avg_reward = np.mean([r['total_reward'] for r in test_results])
                    avg_energy = np.mean([r['total_energy'] for r in test_results])
                    completion_rate = np.mean([r['completed'] for r in test_results])
                    
                    # 综合评分（可以根据需要调整权重）
                    score = avg_reward * 0.6 + completion_rate * 100 * 0.3 - avg_energy * 0.1
                    
                    result = {
                        'trial': i + 1,
                        'config': config.copy(),
                        'score': score,
                        'avg_reward': avg_reward,
                        'avg_energy': avg_energy,
                        'completion_rate': completion_rate,
                        'best_training_reward': monitor.best_avg_reward,
                        'training_episodes': len(monitor.episode_rewards)
                    }
                    
                    self.results.append(result)
                    
                    print(f"评估结果:")
                    print(f"  综合评分: {score:.2f}")
                    print(f"  平均奖励: {avg_reward:.2f}")
                    print(f"  平均能耗: {avg_energy:.2f} J")
                    print(f"  完成率: {completion_rate:.1%}")
                    
                    # 更新最佳配置
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()
                        print(f"  *** 新的最佳配置! 评分: {score:.2f} ***")
                    
                    # 保存中间结果
                    self._save_results()
                
            except Exception as e:
                print(f"试验 {i+1} 失败: {str(e)}")
                continue
        
        print(f"\n=== 调优完成 ===")
        print(f"最佳评分: {best_score:.2f}")
        print(f"最佳配置: {json.dumps(best_config, indent=2, ensure_ascii=False)}")
        
        # 保存最终结果
        self._save_final_results(best_config, best_score)
        
        return self.results, best_config, best_score
    
    def _save_results(self):
        """保存中间结果"""
        results_file = os.path.join(self.results_dir, 'tuning_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def _save_final_results(self, best_config, best_score):
        """保存最终结果"""
        final_results = {
            'search_type': self.search_type,
            'total_trials': len(self.results),
            'best_score': best_score,
            'best_config': best_config,
            'all_results': self.results,
            'completed_at': datetime.now().isoformat()
        }
        
        final_file = os.path.join(self.results_dir, 'final_results.json')
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"最终结果已保存到: {final_file}")


def quick_hyperparameter_search():
    """快速超参数搜索"""
    # 基础配置
    base_config = {
        'num_uavs': 3,
        'num_users': 6,
        'max_episodes': 300,  # 减少训练轮数以加快搜索
        'max_steps_per_episode': 100,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005,
        'max_distance': 100.0,
        'print_interval': 50,
        'eval_interval': 100
    }
    
    # 关键超参数搜索范围
    param_ranges = {
        'lr_actor': [5e-5, 1e-4, 2e-4],      # Actor学习率
        'lr_critic': [5e-4, 1e-3, 2e-3],     # Critic学习率
        'batch_size': [32, 64, 128],          # 批大小
        'tau': [0.001, 0.005, 0.01]          # 软更新系数
    }
    
    print("开始快速超参数搜索...")
    tuner = HyperparameterTuner(base_config, param_ranges, search_type='grid')
    results, best_config, best_score = tuner.run_tuning()
    
    return results, best_config, best_score


def comprehensive_hyperparameter_search():
    """全面的超参数搜索"""
    # 基础配置
    base_config = {
        'num_uavs': 3,
        'num_users': 6,
        'max_episodes': 500,
        'max_steps_per_episode': 150,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005,
        'max_distance': 100.0,
        'print_interval': 50,
        'eval_interval': 100
    }
    
    # 更广泛的参数搜索范围
    param_ranges = {
        'lr_actor': [1e-5, 1e-3],             # 连续范围
        'lr_critic': [1e-4, 5e-3],            # 连续范围
        'batch_size': [32, 64, 128, 256],     # 离散选择
        'tau': [0.001, 0.02],                 # 连续范围
        'gamma': [0.95, 0.99],                # 连续范围
        'max_steps_per_episode': [100, 150, 200]  # 离散选择
    }
    
    print("开始全面超参数搜索...")
    tuner = HyperparameterTuner(base_config, param_ranges, search_type='random')
    results, best_config, best_score = tuner.run_tuning(max_trials=15)
    
    return results, best_config, best_score


def main():
    """主函数"""
    print("=== 多无人机DDPG超参数调优 ===")
    print("1. 快速搜索 (网格搜索，较少参数组合)")
    print("2. 全面搜索 (随机搜索，更多参数组合)")
    
    choice = input("请选择搜索类型 (1/2): ").strip()
    
    if choice == '1':
        results, best_config, best_score = quick_hyperparameter_search()
    elif choice == '2':
        results, best_config, best_score = comprehensive_hyperparameter_search()
    else:
        print("无效选择，使用快速搜索...")
        results, best_config, best_score = quick_hyperparameter_search()
    
    print(f"\n=== 调优总结 ===")
    print(f"最佳配置评分: {best_score:.2f}")
    print("最佳超参数配置:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # 保存最佳配置到单独文件
    best_config_file = f'best_hyperparameters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(best_config_file, 'w') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n最佳配置已保存到: {best_config_file}")
    print("您可以使用这个配置来训练最终模型。")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()

