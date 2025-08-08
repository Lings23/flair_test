from datetime import datetime
import torch
from .observer import Observer

class ServerObserver(Observer):
    def __init__(self, config, observer_config):
        """
        Observer of Server to push model state to victoria metrics
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        :param server_id: server id
        :type server_id: int
        """
        super(ServerObserver, self).__init__(config, observer_config)
        self.name = self.observer_config.server_name 
        self.server_id = self.observer_config.server_id
        self.num_rounds = self.config.ROUNDS
        self.num_clients = self.config.NUMBER_OF_CLIENTS
        self.type = self.observer_config.server_type
        self.metric_labels = { 
            "accuracy": "",
            "recall" : ",target={}",
            "precision" : ",target={}",
            "shap_pos": ",target={},source={}",
            "shap_neg": ",target={},source={}",
            "shap_mean": ",target={},source={}",
            "shap_pos_mean": ",target={},source={}",
            "shap_neg_mean": ",target={},source={}"
        }
        self.metrics = ["accuracy", "recall", "precision", "shap_pos", "shap_neg", "shap_mean", "shap_pos_mean", "shap_neg_mean"]
    
    def get_labels(self):
        """
        Creates Victoria Metrics meta data string
        """
        return "server_id={},test={},type={},experiment_type={},experiment_id={},poisoned_clients={},num_of_rounds={},num_clients={},dataset_type={},round={}".format(
            self.server_id,
            self.test,
            self.type,
            self.experiment_type,
            self.experiment_id,
            self.poisoned_clients,
            self.num_rounds,
            self.num_clients,
            self.dataset_type,
            self.rounds
        )

    def get_datastr(self, recall, precision, accuracy, shap_pos, shap_neg, shap_mean, shap_pos_mean, shap_neg_mean, timestamp=None):
        """
        Creates data string for victoria metrics
        :param accuracy: test accuracy 
        :type  accuracy: float
        :param recall: test recall matrix
        :type  recall: Tensor
        :param precision: test precision matrix
        :type  precision: Tensor
        :param shap_pos: positive SHAP value matrix
        :type  shap_pos: Tensor
        :param shap_neg: negativ SHAP value matrix
        :type  shap_neg: Tensor
        :param shap_mean: mean of SHAP values
        :type  shap_mean: Tensor
        """
        if not timestamp:
            timestamp = int(datetime.timestamp(datetime.now()))
        data = []
        labels = self.get_labels()
        datastr = "{},{} {} {}"
        data.append(datastr.format(self.name, labels, "accuracy=%f"%(accuracy), timestamp))
        for i in range(self.config.NUMBER_TARGETS): 
            data.append(datastr.format(self.name, labels + self.metric_labels["recall"].format(i), "recall=%f"%(recall[i]), timestamp))
            data.append(datastr.format(self.name, labels + self.metric_labels["precision"].format(i), "precision=%f"%(precision[i]), timestamp))
            for j in range(self.config.NUMBER_TARGETS): 
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_pos"].format(i, j), "shap_pos=%f"%(shap_pos[i][j]), timestamp))
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_neg"].format(i, j), "shap_neg=%f"%(shap_neg[i][j]), timestamp))
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_mean"].format(i, j), "shap_mean=%f"%(shap_mean[i][j]), timestamp))
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_pos_mean"].format(i, j), "shap_pos_mean=%f"%(shap_pos_mean[i][j]), timestamp))
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_neg_mean"].format(i, j), "shap_neg_mean=%f"%(shap_neg_mean[i][j]), timestamp))
        return data
    
    def push_metrics(self, recall, precision, accuracy, shap_pos, shap_neg, shap_mean, shap_pos_mean, shap_neg_mean, timestamp=None):
        """
        Push SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        and test metrics like accuracy, precision and recall to victoria metrics
        :param accuracy: test accuracy 
        :type  accuracy: float
        :param recall: test recall matrix
        :type  recall: Tensor
        :param precision: test precision matrix
        :type  precision: Tensor
        :param shap_pos: positive SHAP value matrix
        :type  shap_pos: Tensor
        :param shap_neg: negativ SHAP value matrix
        :type  shap_neg: Tensor
        :param shap_mean: mean of SHAP values
        :type  shap_mean: Tensor
        """
        data = self.get_datastr(recall, precision, accuracy, shap_pos, shap_neg, shap_mean, shap_pos_mean, shap_neg_mean, timestamp)
        for d in data:
            self.push_data(d)
        print("Successfully pushed server data to victoria metrics")
    
    def update_config(self, config, observer_config):
        """
        Update client observer configurations 
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        """
        super().update_config(config, observer_config)
        self.name = self.observer_config.server_name 
        self.num_rounds = self.config.ROUNDS
        self.num_clients = self.config.NUMBER_OF_CLIENTS
        self.type = self.observer_config.server_type        
                
        
    