from abc import abstractmethod
class Dataset(): 
    
    def __init__(self, config):
        self.config = config
        self.train_dataloader, self.train_dataset = self.load_train_data()
        self.test_dataloader, self.test_dataset  = self.load_test_data()
    
    @abstractmethod
    def load_train_data(self):
        """
        Loads & returns the training dataloader and dataset.

        :return: torch.utils.data.Dataloader, torchvision.datasets.Dataset
        """
        raise NotImplementedError("load_train_dataloader() isn't implemented")
        
    @abstractmethod
    def load_test_data(self):
        """
        Loads & returns the test dataloader and dataset. 

        :return: torch.utils.data.Dataloader, torchvision.datasets.Dataset
        """
        raise NotImplementedError("load_test_dataloader() isn't implemented")