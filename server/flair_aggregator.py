import torch
import numpy as np
import copy
from typing import List, Dict, Tuple
import torch.nn.functional as F

class FLAIRAggregator:
    """
    FLAIR: Federated Learning Adversary Identification and Reputation
    基于声誉机制的鲁棒聚合算法
    """
    
    def __init__(self, config):
        self.config = config
        self.num_clients = getattr(config, 'NUMBER_OF_CLIENTS', 10)
        self.clients_per_round = getattr(config, 'CLIENTS_PER_ROUND', 5)
        self.c_max = getattr(config, 'C_MAX', 1)  # 每轮惩罚客户端数量
        self.mu = getattr(config, 'FLAIR_MU', 0.9)  # 声誉衰减因子
        
        # 初始化声誉分数
        self.reputation_scores = np.zeros(self.num_clients)
        self.global_reference_direction = None
        self.previous_global_model = None  # 上一轮全局模型参数
        
    def flair_aggregate(self, parameters: List[Dict], client_indices: List[int] = None) -> Dict:
        """
        FLAIR聚合算法
        Args:
            parameters: 客户端模型参数列表（训练后的参数）
            client_indices: 客户端索引列表（用于追踪声誉）
        Returns:
            聚合后的模型参数
        """
        if client_indices is None:
            client_indices = list(range(len(parameters)))
            
        # 1. 计算客户端更新（相对于上一轮全局模型）
        client_updates = self._compute_client_updates(parameters)
        
        # 2. 计算Flip-Score
        flip_scores = self._compute_flip_scores(client_updates)
        
        # 3. 筛选异常客户端
        punished_clients, rewarded_clients = self._identify_anomalous_clients(flip_scores)
        
        # 4. 更新声誉分数
        self._update_reputation_scores(client_indices, punished_clients, rewarded_clients)
        
        # 5. 计算聚合权重
        aggregation_weights = self._compute_aggregation_weights(client_indices)
        
        # 6. 加权聚合
        aggregated_params = self._weighted_aggregate(parameters, aggregation_weights)
        
        # 7. 更新全局参考方向
        self._update_global_reference_direction(aggregated_params)
        
        return aggregated_params
    
    def _compute_client_updates(self, parameters: List[Dict]) -> List[Dict]:
        """
        计算客户端更新（相对于上一轮全局模型）
        客户端更新 = 客户端训练后参数 - 上一轮全局模型参数
        """
        if self.previous_global_model is None:
            # 第一轮，假设所有客户端从相同的初始模型开始训练
            # 因此客户端更新就是他们训练后的参数
            return parameters
        
        updates = []
        for client_params in parameters:
            update = {}
            for name in client_params.keys():
                # 正确的更新计算：客户端参数 - 上一轮全局模型参数
                update[name] = client_params[name] - self.previous_global_model[name]
            updates.append(update)
        
        return updates
    
    def _compute_flip_scores(self, client_updates: List[Dict]) -> List[float]:
        """
        计算Flip-Score（方向反转分数）
        FS_i = Σ_j (ΔLM_i(j)^2 * 1[sign(ΔLM_i(j)) ≠ sign(s_g(j))])
        """
        if self.global_reference_direction is None:
            # 第一轮，所有客户端Flip-Score为0
            return [0.0] * len(client_updates)
        
        flip_scores = []
        for client_update in client_updates:
            score = 0.0
            for name in client_update.keys():
                client_update_tensor = client_update[name].flatten()
                ref_direction_tensor = self.global_reference_direction[name].flatten()
                
                # 计算方向不一致的部分
                sign_mismatch = torch.sign(client_update_tensor) != torch.sign(ref_direction_tensor)
                score += torch.sum((client_update_tensor ** 2) * sign_mismatch.float()).item()
            
            flip_scores.append(score)
        
        return flip_scores
    
    def _identify_anomalous_clients(self, flip_scores: List[float]) -> Tuple[List[int], List[int]]:
        """
        识别异常客户端
        惩罚Flip-Score最高和最低的前c_max个客户端
        """
        # 按Flip-Score排序
        sorted_indices = np.argsort(flip_scores)
        
        # 惩罚Flip-Score最高和最低的客户端
        punished_clients = []
        rewarded_clients = []
        
        # 惩罚Flip-Score最高的c_max个客户端（方向反转攻击）
        punished_clients.extend(sorted_indices[-self.c_max:])
        
        # 惩罚Flip-Score最低的c_max个客户端（伪装型攻击）
        punished_clients.extend(sorted_indices[:self.c_max])
        
        # 奖励中间的客户端
        middle_start = self.c_max
        middle_end = len(sorted_indices) - self.c_max
        rewarded_clients.extend(sorted_indices[middle_start:middle_end])
        
        return punished_clients, rewarded_clients
    
    def _update_reputation_scores(self, client_indices: List[int], 
                                punished_clients: List[int], 
                                rewarded_clients: List[int]):
        """
        更新声誉分数
        RS(i, t) = μ * RS(i, t-1) + W(i, t)
        """
        m = len(client_indices)
        
        for i, client_idx in enumerate(client_indices):
            if i in punished_clients:
                # 惩罚权重
                weight = -(1 - 2 * self.c_max / m)
            elif i in rewarded_clients:
                # 奖励权重
                weight = 2 * self.c_max / m
            else:
                # 中性权重
                weight = 0.0
            
            # 更新声誉分数
            self.reputation_scores[client_idx] = (
                self.mu * self.reputation_scores[client_idx] + weight
            )
    
    def _compute_aggregation_weights(self, client_indices: List[int]) -> List[float]:
        """
        计算聚合权重（使用softmax归一化）
        WR_i = e^(RS(i)) / Σ_j e^(RS(j))
        """
        # 获取参与客户端的声誉分数
        reputation_subset = [self.reputation_scores[i] for i in client_indices]
        
        # Softmax归一化
        exp_scores = np.exp(reputation_subset)
        total_exp = np.sum(exp_scores)
        
        if total_exp == 0:
            # 如果所有声誉分数都为0，使用均匀权重
            weights = [1.0 / len(client_indices)] * len(client_indices)
        else:
            weights = exp_scores / total_exp
        
        return weights
    
    def _weighted_aggregate(self, parameters: List[Dict], weights: List[float]) -> Dict:
        """
        加权聚合
        GM(t+1) = Σ_i WR_i * LM_i(t+1)
        """
        aggregated_params = {}
        
        for name in parameters[0].keys():
            weighted_sum = torch.zeros_like(parameters[0][name])
            for i, (client_params, weight) in enumerate(zip(parameters, weights)):
                weighted_sum += weight * client_params[name]
            aggregated_params[name] = weighted_sum
        
        return aggregated_params
    
    def _update_global_reference_direction(self, aggregated_params: Dict):
        """
        更新全局参考方向
        s_g(t+1) = sign(ΔGM(t+1))
        其中 ΔGM(t+1) = GM(t+1) - GM(t)
        """
        if self.previous_global_model is not None:
            # 计算全局模型更新
            global_update = {}
            for name in aggregated_params.keys():
                global_update[name] = aggregated_params[name] - self.previous_global_model[name]
            
            # 更新参考方向
            self.global_reference_direction = {}
            for name in global_update.keys():
                self.global_reference_direction[name] = torch.sign(global_update[name])
        
        # 保存当前聚合结果作为下一轮的全局模型
        self.previous_global_model = copy.deepcopy(aggregated_params)
    
    def get_reputation_scores(self) -> np.ndarray:
        """获取当前声誉分数"""
        return self.reputation_scores.copy()
    
    def get_client_weights(self, client_indices: List[int]) -> List[float]:
        """获取指定客户端的聚合权重"""
        return self._compute_aggregation_weights(client_indices)
    
    def reset_reputation_scores(self):
        """重置声誉分数"""
        self.reputation_scores = np.zeros(self.num_clients)
        self.global_reference_direction = None
        self.previous_global_model = None 