from datetime import datetime
from .observer import Observer
import torch

class ClientObserver(Observer):
    def __init__(self, config, observer_config, client_id, poisoned, dataset_size):
        """
        Observer of Client to push model state to victoria metrics
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        :param client_id: client id
        :type config: int
        :param poisoned: poisoned toggle
        :type poisoned: boolean 
        :param dataset_size: size of dataset
        :type poisoned: int
        """
        super(ClientObserver, self).__init__(config, observer_config)
        self.name = self.observer_config.client_name 
        self.client_id = client_id
        self.poisoned = poisoned
        self.poisoned_data = self.config.DATA_POISONING_PERCENTAGE if poisoned else 0
        self.num_epoch = self.config.N_EPOCHS
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.num_clients = self.config.NUMBER_OF_CLIENTS
        self.num_poisoned_clients = self.config.POISONED_CLIENTS
        self.dataset_size = dataset_size
        self.type = self.observer_config.client_type
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
        self.metrics = ["accuracy", "recall", "precision", "shap_pos", "shap_neg", "shap_mean"]
    
    def set_poisoned(self, poisoned):
        self.poisoned = poisoned
    
    def get_labels(self):
        """
        Creates Victoria Metrics meta data string
        """
        return "client_id={},test={},poisoned={},poisoned_data={},poisoned_clients={},dataset_size={},type={},experiment_type={},experiment_id={},poisoned_clients={},num_of_epochs={},batch_size={},num_clients={},dataset_type={},round={}".format(
            self.client_id,
            self.test,
            self.poisoned,
            self.poisoned_data,
            self.poisoned_clients,
            self.dataset_size,
            self.type,
            self.experiment_type,
            self.experiment_id,
            self.poisoned_clients,
            self.num_epoch,
            self.batch_size,
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
    
    def push_metrics(self,recall, precision, accuracy, shap_pos, shap_neg, shap_mean, shap_pos_mean, shap_neg_mean, timestamp=None):
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
        print("Successfully pushed client data to victoria metrics")
    
    def update_config(self, config, observer_config):
        """
        Update client observer configurations 
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        """
        super().update_config(config, observer_config)
        self.name = self.observer_config.client_name 
        self.poisoned_data = self.config.DATA_POISONING_PERCENTAGE
        self.num_epoch = self.config.N_EPOCHS
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.num_clients = self.config.NUMBER_OF_CLIENTS
        self.type = self.observer_config.client_type

        
        
        
                
                
        
        
    