import os
import flwr as fl
from flwr.common.typing import Scalar, Parameters
from flwr.common.parameter import weights_to_parameters
import ray
import torch
import torchvision
import torch.nn as nn
from torch.nn import Module, GroupNorm
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from dataset_utils import getCIFAR10, getSTL10, do_fl_partitioning, get_dataloader_double, get_dataloader
from torchvision.models.resnet import resnet18
import math
import argparse

# CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#     'dog', 'frog', 'horse', 'ship', 'truck']

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example

global_lr = None
momentum = None
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.resnet = resnet18(weights='ResNet18_Weights.DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x


# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str, rnd):
    """Train the network on the training set."""
    client_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    lr = global_lr*((0.998)**(float(rnd)-1))
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            client_loss += loss
        scheduler.step()
    return client_loss

# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for j, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, fed_dir_data2: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.fed_dir2 = Path(fed_dir_data2)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        
        # instantiate model

        self.net = Net()

        # determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)

    def get_loss(self, parameters):
        pass

    def fit(self, parameters, config):

        # print(f"fit() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        #num_workers = len(ray.worker.get_resource_ids()["CPU"])
        num_workers = len(ray.get_runtime_context().get_assigned_resources())
        trainloader = get_dataloader_double(
            self.fed_dir,
            self.fed_dir2,
            self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
        )
        # send model to device
        self.net.to(self.device)

        # train
        client_loss=train(self.net, trainloader, epochs=int(config["epochs"]), device=self.device, rnd=config["epoch_global"])

        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), float(client_loss)

    def evaluate(self, parameters, config):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        #num_workers = len(ray.worker.get_resource_ids()["CPU"])
        num_workers = len(ray.get_runtime_context().get_assigned_resources())
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)  

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(10),
        "batch_size": str(50),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)
        
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def original_get_cifar_model(num_classes: int = 10) -> Module:
    model: ResNet = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    return model


def get_cifar_model(num_classes: int = 10) -> Module:
    model = Net()
    return model

def get_initial_parameters(num_classes: int = 10) -> Parameters:
    model = get_cifar_model(num_classes)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = weights_to_parameters(weights)

    return parameters


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 4. Starts a Ray-based simulation where a % of clients are sample each round.
# 5. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--num', type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    option = parser.parse_args()
    global_lr = option.lr
    momentum = option.momentum

    if option.device == "gpu":
        client_resources = {"num_gpus":1}  # each client will get allocated 1 CPUs
    elif option.device == "cpu":
        client_resources = {"num_cpus": 1}  # each client will get allocated 1 CPUs
    else:
        print('input device, ex) cpu or gpu')
        exit(1)

    if option.num is None:
        print('input device number, ex) 0,1,...')
        exit(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.num

    print("====================================")
    print("[main]cuda,",torch.cuda.is_available())
    print("====================================")

    pool_size = 100  # number of dataset partions (= number of total clients)
    # download CIFAR10 dataset
    train_path, testset = getCIFAR10()
    train_path2, _ = getSTL10()
    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated: in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )
    
    fed_dir2 = do_fl_partitioning(
        train_path2, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.0
    )

    def get_initial_parameters(num_classes):
        model = Net()
        weights = [val.cpu().numpy() for _,val in model.state_dict().items()]
        return weights

    # configure the strategy
    initial_parameters: Parameters = get_initial_parameters(10)    
 
    strategy = fl.server.strategy.FedAvg(
        fraction_fit= 0.1,
        min_fit_clients=int(0.3 * pool_size),
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
        initial_parameters=get_initial_parameters(10)
     )

    strategy.initial_parameters = initial_parameters

     
    def client_fn(cid: str):
        # create a single client instance
        return CifarRayClient(cid, fed_dir, fed_dir2)
    
    # (optional) specify ray config
    ray_config = {
        "include_dashboard": False,
        "num_gpus": 1,
        "num_cpus": 1,
    }

    # start simulation
    output = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=400,
        strategy=strategy,
        ray_init_args=ray_config,
    )
    # print("output:", output.metrics_centralized)
    # print("type:", type(output.metrics_centralized))
    import json

    with open ('{}_{}.json'.format(global_lr, momentum), 'w') as f:
        json.dump(output.metrics_centralized, f)

