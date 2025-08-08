import torch
import numpy as np
import copy
from typing import List, Dict, Tuple
import torch.nn.functional as F

class RobustAggregator:
    """
    鲁棒聚合器，实现多种防御拜占庭攻击的聚合算法
    """
    
    def __init__(self, config):
        self.config = config
        self.f = 0  # 拜占庭客户端数量上限
        
    def krum_aggregate(self, parameters: List[Dict], client_scores: List[float] = None) -> Dict:
        """
        Krum 聚合算法
        Args:
            parameters: 客户端模型参数列表
            client_scores: 客户端信任分数列表（可选）
        Returns:
            聚合后的模型参数
        """
        if len(parameters) <= 2 * self.f + 1:
            print("Warning: 客户端数量不足以使用Krum算法")
            return self.fedavg_aggregate(parameters)
            
        # 计算每对客户端之间的距离
        distances = self._compute_distances(parameters)
        
        # 为每个客户端计算分数
        scores = []
        for i in range(len(parameters)):
            # 选择最近的 n-f-2 个客户端
            distances_i = distances[i].copy()
            distances_i[i] = float('inf')  # 排除自己
            sorted_indices = np.argsort(distances_i)
            selected_indices = sorted_indices[:len(parameters) - self.f - 2]
            
            # 计算分数
            score = sum(distances_i[j] for j in selected_indices)
            scores.append(score)
        
        # 选择分数最小的客户端
        best_client = np.argmin(scores)
        print(f"Krum选择客户端 {best_client}，分数: {scores[best_client]:.4f}")
        
        return parameters[best_client]
    
    def fltrust_aggregate(self, parameters: List[Dict], server_model: Dict, 
                         trust_scores: List[float] = None) -> Dict:
        """
        FLTrust 聚合算法
        Args:
            parameters: 客户端模型参数列表
            server_model: 服务器模型参数
            trust_scores: 客户端信任分数列表（可选）
        Returns:
            聚合后的模型参数
        """
        if not trust_scores:
            trust_scores = [1.0] * len(parameters)
        
        # 计算每个客户端更新与服务器更新的相似度
        similarities = []
        for client_params in parameters:
            similarity = self._compute_similarity(client_params, server_model)
            similarities.append(similarity)
        
        # 归一化信任分数
        total_trust = sum(trust_scores)
        if total_trust > 0:
            normalized_trust = [score / total_trust for score in trust_scores]
        else:
            normalized_trust = [1.0 / len(parameters)] * len(parameters)
        
        # 加权聚合
        aggregated_params = {}
        for name in parameters[0].keys():
            weighted_sum = torch.zeros_like(parameters[0][name])
            for i, client_params in enumerate(parameters):
                weight = normalized_trust[i] * similarities[i]
                weighted_sum += weight * client_params[name]
            aggregated_params[name] = weighted_sum
        
        return aggregated_params
    
    def _compute_distances(self, parameters: List[Dict]) -> np.ndarray:
        """
        计算客户端模型参数之间的欧几里得距离
        """
        n_clients = len(parameters)
        distances = np.zeros((n_clients, n_clients))
        
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                distance = self._compute_model_distance(parameters[i], parameters[j])
                distances[i][j] = distance
                distances[j][i] = distance
                
        return distances
    
    def _compute_model_distance(self, params1: Dict, params2: Dict) -> float:
        """
        计算两个模型参数之间的欧几里得距离
        """
        total_distance = 0.0
        total_params = 0
        
        for name in params1.keys():
            diff = params1[name] - params2[name]
            distance = torch.norm(diff).item()
            total_distance += distance ** 2
            total_params += diff.numel()
        
        return np.sqrt(total_distance)
    
    def _compute_similarity(self, client_params: Dict, server_params: Dict) -> float:
        """
        计算客户端更新与服务器更新的余弦相似度
        """
        client_update = {}
        server_update = {}
        
        # 计算更新（这里简化处理，实际应该基于上一轮模型）
        for name in client_params.keys():
            client_update[name] = client_params[name]
            server_update[name] = server_params[name]
        
        # 计算余弦相似度
        similarity = self._cosine_similarity(client_update, server_update)
        return max(0, similarity)  # 确保非负
    
    def _cosine_similarity(self, params1: Dict, params2: Dict) -> float:
        """
        计算两个参数字典的余弦相似度
        """
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for name in params1.keys():
            p1 = params1[name].flatten()
            p2 = params2[name].flatten()
            
            dot_product += torch.dot(p1, p2).item()
            norm1 += torch.norm(p1).item() ** 2
            norm2 += torch.norm(p2).item() ** 2
        
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def fedavg_aggregate(self, parameters: List[Dict]) -> Dict:
        """
        联邦平均聚合（FedAvg）
        """
        new_params = {}
        for name in parameters[0].keys():
            new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)
        return new_params
    
    def median_aggregate(self, parameters: List[Dict]) -> Dict:
        """
        中位数聚合
        """
        new_params = {}
        for name in parameters[0].keys():
            stacked_params = torch.stack([param[name] for param in parameters])
            new_params[name] = torch.median(stacked_params, dim=0)[0]
        return new_params
    
    def trimmed_mean_aggregate(self, parameters: List[Dict], trim_ratio: float = 0.1) -> Dict:
        """
        截断均值聚合
        """
        new_params = {}
        n_clients = len(parameters)
        trim_count = int(n_clients * trim_ratio)
        
        for name in parameters[0].keys():
            stacked_params = torch.stack([param[name] for param in parameters])
            sorted_params, _ = torch.sort(stacked_params, dim=0)
            trimmed_params = sorted_params[trim_count:-trim_count] if trim_count > 0 else sorted_params
            new_params[name] = torch.mean(trimmed_params, dim=0)
        
        return new_params 