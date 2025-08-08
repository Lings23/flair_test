import random
import os
from pathlib import Path
import torch

def set_rounds(client_plane, server, rounds):
    client_plane.set_rounds(rounds)
    server.set_rounds(rounds)
    
def update_configs(client_plane, server, config, observer_config):
    client_plane.update_config(config, observer_config)
    server.update_config(config, observer_config)
    
def run_round(client_plane, server, rounds):
    # Federated Learning Round 
    set_rounds(client_plane, server, rounds)
    client_plane.update_clients(server.get_nn_parameters())
    selected_clients = server.select_clients()
    client_parameters = client_plane.train_selected_clients(selected_clients)
    server.aggregate_model(client_parameters)
    
def run_round_with(selected_clients, nn_params, client_plane, server, rounds):
    # Federated Learning Round with preselected nn_params and clients
    set_rounds(client_plane, server, rounds)
    client_plane.update_clients(server.get_nn_parameters())
    client_parameters = client_plane.train_selected_clients(selected_clients)
    server.aggregate_model(client_parameters)
    
def select_random_clean(client_plane, config, n):
    if len(client_plane.poisoned_clients) == config.NUMBER_OF_CLIENTS: 
        return []
    clean = [x for x in range(config.NUMBER_OF_CLIENTS) if x not in client_plane.poisoned_clients]
    random.shuffle(clean)
    indices = clean[:n]
    return indices

def select_poisoned(client_plane, n):
    if len(client_plane.poisoned_clients) > 0:
        return client_plane.poisoned_clients[:n]
    return []

def train_client(client_plane, rounds, idx): 
    client_plane.clients[idx].train(rounds)
    client_plane.clients[idx].push_metrics()

def print_posioned_target(client_plane, idx):
    client = client_plane.clients[idx]
    print(client.train_dataloader.dataset.dataset.targets[client.poisoned_indices][0])

def create_default_model(config):
    default_model_path = os.path.join(config.TEMP, 'models', "{}.model".format(config.MODELNAME))
    net = config.NETWORK()
    Path(os.path.dirname(default_model_path)).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), default_model_path)
    print("default model saved to:{}".format(os.path.dirname(default_model_path)))
    