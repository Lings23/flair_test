import torch, torchvision
from .dataset import Dataset

class CIFAR10Dataset(Dataset): 
    
    def __init__(self, config):
        super(CIFAR10Dataset, self).__init__(config)
        self.labels = self.config.CIFAR10_LABELS
        
    def load_train_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor(),torchvision.transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_dataset = torchvision.datasets.CIFAR10(
            self.config.CIFAR10_DATASET_PATH, 
            train=True, download=True,
            transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE_TRAIN, 
            shuffle=True)
        
        print("CIFAR10 training data loaded.")
        return train_loader, train_dataset
    
    def load_test_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor(),torchvision.transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        test_dataset = torchvision.datasets.CIFAR10(
            self.config.CIFAR10_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE_TEST, 
            shuffle=False)
        
        print("CIFAR10 test data loaded.")
        return test_loader, test_dataset
