#!/usr/bin/env python3
"""
鲁棒聚合算法使用示例

本脚本演示如何使用Krum和FLTrust等鲁棒聚合算法来防御拜占庭攻击。
"""

import sys
import os
sys.path.append('.')

from federated_learning.configuration import Configuration
from federated_learning.server.model_aggregator import ModelAggregator
import torch
import numpy as np

def main():
    """主函数"""
    print("=== 鲁棒聚合算法演示 ===\n")
    
    # 1. 创建配置
    config = Configuration()
    config.NUMBER_OF_CLIENTS = 10
    config.CLIENTS_PER_ROUND = 5
    config.BYZANTINE_F = 2  # 假设有2个拜占庭客户端
    config.AGGREGATION_METHOD = 'krum'  # 可以改为 'fltrust', 'median', 'trimmed_mean'
    
    print(f"配置信息:")
    print(f"- 总客户端数: {config.NUMBER_OF_CLIENTS}")
    print(f"- 每轮参与客户端数: {config.CLIENTS_PER_ROUND}")
    print(f"- 拜占庭客户端数: {config.BYZANTINE_F}")
    print(f"- 聚合方法: {config.AGGREGATION_METHOD}")
    
    # 2. 创建聚合器
    aggregator = ModelAggregator(config)
    
    # 3. 创建模拟的客户端参数
    print("\n创建模拟客户端参数...")
    client_parameters = create_mock_parameters(config)
    
    # 4. 执行聚合
    print(f"\n使用 {config.AGGREGATION_METHOD} 方法进行聚合...")
    
    if config.AGGREGATION_METHOD == 'fltrust':
        # FLTrust需要服务器模型
        server_model = create_server_model()
        result = aggregator.fltrust_aggregate(client_parameters, server_model)
    elif config.AGGREGATION_METHOD == 'krum':
        result = aggregator.krum_aggregate(client_parameters)
    elif config.AGGREGATION_METHOD == 'median':
        result = aggregator.median_aggregate(client_parameters)
    elif config.AGGREGATION_METHOD == 'trimmed_mean':
        result = aggregator.trimmed_mean_aggregate(client_parameters, config.TRIM_RATIO)
    else:
        result = aggregator.fedavg_aggregate(client_parameters)
    
    print(f"聚合完成！结果参数数量: {len(result)}")
    
    # 5. 分析结果
    analyze_results(client_parameters, result, config)

def create_mock_parameters(config):
    """创建模拟的客户端模型参数"""
    parameters_list = []
    
    # 创建正常客户端的参数
    for i in range(config.CLIENTS_PER_ROUND - config.BYZANTINE_F):
        params = {
            'layer1.weight': torch.randn(64, 784) * 0.1,
            'layer1.bias': torch.randn(64) * 0.1,
            'layer2.weight': torch.randn(10, 64) * 0.1,
            'layer2.bias': torch.randn(10) * 0.1
        }
        parameters_list.append(params)
    
    # 创建拜占庭客户端的参数（异常值）
    for i in range(config.BYZANTINE_F):
        params = {
            'layer1.weight': torch.randn(64, 784) * 10.0,  # 异常大的参数
            'layer1.bias': torch.randn(64) * 10.0,
            'layer2.weight': torch.randn(10, 64) * 10.0,
            'layer2.bias': torch.randn(10) * 10.0
        }
        parameters_list.append(params)
    
    return parameters_list

def create_server_model():
    """创建服务器模型参数"""
    return {
        'layer1.weight': torch.randn(64, 784) * 0.1,
        'layer1.bias': torch.randn(64) * 0.1,
        'layer2.weight': torch.randn(10, 64) * 0.1,
        'layer2.bias': torch.randn(10) * 0.1
    }

def analyze_results(client_parameters, aggregated_result, config):
    """分析聚合结果"""
    print("\n=== 结果分析 ===")
    
    # 计算每个客户端参数的范数
    client_norms = []
    for i, params in enumerate(client_parameters):
        norm = compute_parameter_norm(params)
        client_norms.append(norm)
        client_type = "拜占庭" if i >= len(client_parameters) - config.BYZANTINE_F else "正常"
        print(f"客户端 {i} ({client_type}): 范数 = {norm:.4f}")
    
    # 计算聚合结果的范数
    result_norm = compute_parameter_norm(aggregated_result)
    print(f"\n聚合结果范数: {result_norm:.4f}")
    
    # 计算与正常客户端的平均范数差异
    normal_norms = client_norms[:-config.BYZANTINE_F]
    avg_normal_norm = np.mean(normal_norms)
    norm_diff = abs(result_norm - avg_normal_norm)
    print(f"与正常客户端平均范数差异: {norm_diff:.4f}")
    
    # 评估鲁棒性
    if norm_diff < 1.0:
        print("✓ 聚合结果鲁棒性良好")
    else:
        print("⚠ 聚合结果可能受到攻击影响")

def compute_parameter_norm(params):
    """计算参数范数"""
    total_norm = 0.0
    for name, param in params.items():
        total_norm += torch.norm(param).item() ** 2
    return np.sqrt(total_norm)

if __name__ == "__main__":
    main() 