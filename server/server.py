import torch
import os
import copy

from .model_aggregator import ModelAggregator
from .client_selector import ClientSelector
from ..observer import ServerObserver
from ..utils import CNNHandler

class Server(CNNHandler):
    def __init__(self, config, observer_config, train_dataloader, test_dataloader, shap_util):
        
        super(Server, self).__init__(config, observer_config, train_dataloader, test_dataloader, shap_util)
        self.default_model_path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        self.observer = ServerObserver(config, observer_config)
        self.aggregator = ModelAggregator(config)
        self.selector = ClientSelector()
        self.previous_model = None  # 用于FLTrust
        self.selected_clients = []  # 用于FLAIR追踪客户端
    
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
        self.net.eval()
        
    def select_clients(self):
        selected = self.selector.random_selector(self.config.NUMBER_OF_CLIENTS, self.config.CLIENTS_PER_ROUND)
        self.selected_clients = selected  # 保存选中的客户端用于FLAIR
        return selected

    def aggregate_model(self, client_parameters): 
        """
        根据配置选择聚合方法
        """
        method = getattr(self.config, 'AGGREGATION_METHOD', 'fedavg')
        
        if method == 'flair':
            new_parameters = self.aggregator.flair_aggregate(client_parameters, self.selected_clients)
        elif method == 'krum':
            new_parameters = self.aggregator.krum_aggregate(client_parameters)
        elif method == 'fltrust':
            if self.previous_model is None:
                # 第一轮使用FedAvg
                new_parameters = self.aggregator.fedavg_aggregate(client_parameters)
            else:
                new_parameters = self.aggregator.fltrust_aggregate(
                    client_parameters, 
                    self.previous_model,
                    getattr(self.config, 'TRUST_SCORES', None)
                )
        elif method == 'median':
            new_parameters = self.aggregator.median_aggregate(client_parameters)
        elif method == 'trimmed_mean':
            trim_ratio = getattr(self.config, 'TRIM_RATIO', 0.1)
            new_parameters = self.aggregator.trimmed_mean_aggregate(client_parameters, trim_ratio)
        else:  # 默认使用FedAvg
            new_parameters = self.aggregator.fedavg_aggregate(client_parameters)
        
        # 保存当前模型用于下一轮FLTrust
        self.previous_model = copy.deepcopy(new_parameters)
        
        self.update_nn_parameters(new_parameters)
        if (self.rounds + 1)%50 == 0:
            print(f"Model aggregation in round {self.rounds+1} was successful using {method}")
            
            # 如果是FLAIR，打印声誉分数
            if method == 'flair':
                reputation_scores = self.aggregator.get_flair_reputation_scores()
                print(f"FLAIR声誉分数: {reputation_scores}")
        
    def update_config(self, config, observer_config):
        super().update_config(config, observer_config)
        self.default_model_path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        # 更新聚合器配置
        self.aggregator = ModelAggregator(config)