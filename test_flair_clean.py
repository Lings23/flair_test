#!/usr/bin/env python3
"""
FLAIR算法测试脚本 - 完全无SHAP依赖
严格按照FLAIR文档实现，移除所有SHAP依赖
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入联邦学习相关模块（移除SHAP依赖）
from federated_learning.configuration import Configuration
from federated_learning.observer_configuration import ObserverConfiguration
from federated_learning.utils import experiment_util
from federated_learning import ClientPlane
from federated_learning.server import Server
from federated_learning.nets import MNISTCNN
from federated_learning.dataset import MNISTDataset

def test_flair_clean():
    """完全无SHAP依赖的FLAIR测试"""
    print("=== FLAIR算法测试（完全无SHAP依赖）===\n")
    
    # 创建配置
    config = Configuration()
    observer_config = ObserverConfiguration()
    
    # 基础配置
    config.NUMBER_OF_CLIENTS = 5  # 减少客户端数量以便测试
    config.CLIENTS_PER_ROUND = 3
    config.BATCH_SIZE_TRAIN = 16
    config.BATCH_SIZE_TEST = 100
    config.LEARNING_RATE = 0.01
    config.EPOCHS = 1
    config.ROUNDS = 5  # 减少轮次以便快速测试
    config.N_EPOCHS = 1
    
    # 数据集配置
    config.DATASET = MNISTDataset
    config.MODELNAME = config.MNIST_NAME
    config.NETWORK = MNISTCNN
    
    # 攻击配置
    config.POISONED_CLIENTS = 1  # 1个恶意客户端
    config.DATA_POISONING_PERCENTAGE = 1.0
    config.FROM_LABEL = 5
    config.TO_LABEL = 4
    
    # FLAIR配置
    config.AGGREGATION_METHOD = 'flair'
    config.C_MAX = 1
    config.FLAIR_MU = 0.9
    
    # Victoria Metrics配置（修复VM_URL问题）
    config.VM_URL = None  # 设置为None以避免连接问题
    
    print(f"配置信息:")
    print(f"- 总客户端数: {config.NUMBER_OF_CLIENTS}")
    print(f"- 每轮参与客户端数: {config.CLIENTS_PER_ROUND}")
    print(f"- 恶意客户端数: {config.POISONED_CLIENTS}")
    print(f"- 聚合方法: {config.AGGREGATION_METHOD}")
    
    try:
        # 初始化数据集和组件
        print("\n1. 初始化数据集...")
        data = config.DATASET(config)
        
        print("2. 创建默认模型...")
        experiment_util.create_default_model(config)
        
        print("3. 初始化服务器和客户端...")
        # 不传递shap_util参数
        server = Server(config, observer_config, data.train_dataloader, data.test_dataloader, None)
        client_plane = ClientPlane(config, observer_config, data, None)
        
        print("4. 执行标签翻转攻击...")
        client_plane.poison_clients()
        print(f"恶意客户端索引: {client_plane.poisoned_clients}")
        
        print("5. 检查FLAIR聚合器...")
        if hasattr(server.aggregator, 'flair_aggregator'):
            print("✓ FLAIR聚合器已初始化")
        else:
            print("⚠ FLAIR聚合器未初始化")
            return False
        
        print("6. 执行联邦学习训练...")
        reputation_history = []
        accuracy_history = []
        
        for round_idx in range(config.ROUNDS):
            print(f"  第 {round_idx + 1} 轮...")
            
            # 执行一轮训练
            experiment_util.run_round(client_plane, server, round_idx + 1)
            
            # 检查声誉分数
            if hasattr(server.aggregator, 'get_flair_reputation_scores'):
                reputation_scores = server.aggregator.get_flair_reputation_scores()
                if len(reputation_scores) > 0:
                    print(f"    声誉分数: {[f'{r:.3f}' for r in reputation_scores]}")
                    reputation_history.append(reputation_scores.copy())
                    
                    # 分析恶意客户端检测效果
                    if len(client_plane.poisoned_clients) > 0:
                        malicious_scores = [reputation_scores[i] for i in client_plane.poisoned_clients]
                        normal_scores = [reputation_scores[i] for i in range(len(reputation_scores)) 
                                       if i not in client_plane.poisoned_clients]
                        
                        avg_malicious = np.mean(malicious_scores)
                        avg_normal = np.mean(normal_scores)
                        print(f"    恶意客户端平均声誉: {avg_malicious:.3f}")
                        print(f"    正常客户端平均声誉: {avg_normal:.3f}")
                        print(f"    声誉差异: {avg_normal - avg_malicious:.3f}")
                else:
                    print("    声誉分数: 未初始化")
        
        print("7. 测试模型性能...")
        server.test()
        recall, precision, accuracy = server.analize_test()
        # 召回率/精确率是按类别的张量，打印宏平均
        recall_mean = float(torch.mean(recall).item()) if isinstance(recall, torch.Tensor) else float(recall)
        precision_mean = float(torch.mean(precision).item()) if isinstance(precision, torch.Tensor) else float(precision)
        print(f"最终性能 - 准确率: {accuracy:.3f}, 召回率(宏平均): {recall_mean:.3f}, 精确率(宏平均): {precision_mean:.3f}")
        
        # 8. 可视化声誉分数变化
        if len(reputation_history) > 0:
            reputation_history = np.array(reputation_history)
            
            plt.figure(figsize=(12, 8))
            
            # 声誉分数变化
            plt.subplot(2, 2, 1)
            for client_idx in range(reputation_history.shape[1]):
                client_type = "恶意" if client_idx in client_plane.poisoned_clients else "正常"
                plt.plot(range(1, len(reputation_history) + 1), reputation_history[:, client_idx], 
                         marker='o', label=f'客户端 {client_idx} ({client_type})', linewidth=2)
            
            plt.xlabel('训练轮次')
            plt.ylabel('声誉分数')
            plt.title('FLAIR算法声誉分数变化')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 恶意客户端检测效果
            plt.subplot(2, 2, 2)
            if len(client_plane.poisoned_clients) > 0:
                malicious_reputation = reputation_history[:, client_plane.poisoned_clients]
                normal_reputation = reputation_history[:, [i for i in range(config.NUMBER_OF_CLIENTS) 
                                                         if i not in client_plane.poisoned_clients]]
                
                plt.plot(range(1, len(reputation_history) + 1), np.mean(malicious_reputation, axis=1), 
                         marker='o', label='恶意客户端平均声誉', color='red', linewidth=2)
                plt.plot(range(1, len(reputation_history) + 1), np.mean(normal_reputation, axis=1), 
                         marker='s', label='正常客户端平均声誉', color='blue', linewidth=2)
            
            plt.xlabel('训练轮次')
            plt.ylabel('平均声誉分数')
            plt.title('恶意客户端检测效果')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 声誉分布热力图
            plt.subplot(2, 2, 3)
            im = plt.imshow(reputation_history.T, cmap='RdYlBu_r', aspect='auto')
            plt.colorbar(im)
            plt.xlabel('训练轮次')
            plt.ylabel('客户端索引')
            plt.title('声誉分数热力图')
            plt.yticks(range(config.NUMBER_OF_CLIENTS))
            
            # 声誉差异
            plt.subplot(2, 2, 4)
            if len(client_plane.poisoned_clients) > 0:
                reputation_diff = np.mean(normal_reputation, axis=1) - np.mean(malicious_reputation, axis=1)
                plt.plot(range(1, len(reputation_history) + 1), reputation_diff, 
                         marker='^', color='purple', linewidth=2)
            
            plt.xlabel('训练轮次')
            plt.ylabel('声誉差异')
            plt.title('正常与恶意客户端声誉差异')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flair_clean()
    if success:
        print("✓ FLAIR测试通过（完全无SHAP依赖）")
    else:
        print("✗ FLAIR测试失败") 