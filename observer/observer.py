from ..utils import VMUtil

class Observer(VMUtil):
    """
    Observer to push model states to victoria metrics
    :param config: experiment configurations
    :type config: Configuration
    :param observer_config: observer configurations
    :type observer_config: ObserverConfiguration
    """
    def __init__(self, config, observer_config):
        super(Observer, self).__init__(config)
        self.config = config
        self.observer_config = observer_config
        self.experiment_type = self.observer_config.experiment_type
        self.experiment_id = self.observer_config.experiment_id
        self.poisoned_clients = self.config.POISONED_CLIENTS
        self.test = self.observer_config.test
        self.dataset_type = self.observer_config.dataset_type
        self.rounds = 0
        
    def set_rounds(self, rounds): 
        self.rounds = rounds

    def update_config(self, config, observer_config):
        self.config = config
        self.observer_config = observer_config
        self.experiment_type = self.observer_config.experiment_type
        self.experiment_id = self.observer_config.experiment_id
        self.poisoned_clients = self.config.POISONED_CLIENTS
        self.test = self.observer_config.test
        self.dataset_type = self.observer_config.dataset_type