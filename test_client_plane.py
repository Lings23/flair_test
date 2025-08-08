from .configuration import Configuration
from .dataset import Dataset
from .client import Client
import torch
from numpy.random import default_rng
from .client_plane import ClientPlane


class TestClientPlane(ClientPlane):
    
    def __init__(self, config, observer_config, data, shap_util):
        super(TestClientPlane, self).__init__(config, observer_config, data, shap_util)
        self.config = config
        self.observer_config = observer_config
        self.shap_util = shap_util
        self.train_dataset = data.test_dataset
        self.test_dataset = data.test_dataset
        self.train_dataloader = data.test_dataset
        self.test_dataloader = data.test_dataloader
        self.clients = self.create_clients()
        self.poisoned_clients = []
        self.rounds = 0
    