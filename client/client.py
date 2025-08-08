import torch
from ..observer import ClientObserver
from ..utils import CNNHandler
import copy
import random

class Client(CNNHandler): 
    def __init__(self, config, observer_config,train_dataloader, test_dataloader, shap_util, client_id):
        """
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        :param client_id: client id
        :type observerconfig: int
        :param train_dataloader: Training data loader
        :type train_dataloader: torch.utils.data.DataLoader
        :param test_dataloader: Test data loader
        :type test_dataloader: torch.utils.data.DataLoader
        :param shap_util: utils for shap calculations (optional)
        :type shap_util: SHAPUtil or None
        """
        super(Client, self).__init__(config, observer_config, train_dataloader, test_dataloader, shap_util)
        self.observer = ClientObserver(self.config, self.observer_config, client_id, False, len(train_dataloader.dataset))
        self.client_id = client_id
        
        # label flipping meta data
        self.is_poisoned = False
        self.poisoned_indices = []
        self.poisoning_indices = []
        
    def get_to_poison_indices(self, from_label):
        indices = []
        for idx in self.train_dataloader.dataset.indices: 
            if self.train_dataloader.dataset.dataset.targets[idx] == from_label:
                indices.append(idx)
        return indices
        
    def label_flipping_data(self, from_label, to_label, percentage=1): 
        """
        Label Flipping attack on distributed client 
        :param from_label: label to be flipped
        :type from_label: 
        :param to_label: label flipped to
        :type to_label: 
        """
        indices = self.get_to_poison_indices(from_label)
        last_index = int(len(indices) * percentage)
        self.poisoning_indices = indices
        self.poisoned_indices = indices if percentage == 1 else indices[:last_index]
        self.train_dataloader.dataset.dataset.targets[self.poisoned_indices] = to_label
        self.observer.set_poisoned(True)
        self.is_poisoned = True

    def random_label_flipping_data(self, percentage=1): 
        """
        Label Flipping attack on distributed client with random assignments
        :param from_label: label to be flipped
        :type from_label: 
        :param to_label: label flipped to
        :type to_label: 
        """
        self.targets = copy.deepcopy(self.train_dataloader.dataset.dataset.targets[self.train_dataloader.dataset.indices])
        indices =(self.train_dataloader.dataset.dataset.targets[self.train_dataloader.dataset.indices]).nonzero(as_tuple=False)
        last_index = int(len(indices) * percentage)
        self.poisoned_indices = indices if percentage == 1 else indices[:last_index]
        for idx in self.poisoned_indices:
            self.train_dataloader.dataset.dataset.targets[self.poisoned_indices] = random.choice(range(self.config.NUMBER_TARGETS).remove(self.train_dataloader.dataset.dataset.targets[self.poisoned_indices]))
        self.observer.set_poisoned(True)
        self.is_poisoned = True
        
    def reset_label_flipping_data(self, from_label, percentage=1):
        if self.is_poisoned: 
            self.train_dataloader.dataset.dataset.targets[self.poisoning_indices] = from_label
            self.is_poisoned = False
            self.observer.set_poisoned(False)
            
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
        self.net.eval()
        
    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()
    
