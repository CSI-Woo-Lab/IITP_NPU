import os
import gc
import math
import numpy as np
import wandb
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
import copy
import argparse
import yaml
import time

import flwr as fl
import torch

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.fedavg import FedAvg

# from custom_ultralytics.nn.tasks import DetectionModel
# from custom_ultralytics.models.yolo.detect import DetectionValidator

from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load, torch_safe_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    TQDM,
    emojis,
    clean_url,
    yaml_load,
)
from ultralytics.utils.ops import Profile

# small_yaml = 'configs/small_detection_cfg.yaml'
# large_yaml = 'configs/large_detection_cfg.yaml'
# default_yaml = small_yaml
model_name = 'model'

best_accuracy = 0
global_round = 0

batch_size = 64


def _setup_train(model, data = "configs/keti_fl_dataset_0.yaml"):
    train_kwargs = {
        'data': data,
        'batch': batch_size,
        'epochs': 1,
        'imgsz': 640,
        'verbose': False,
        'val': False,
        'save': False,
    }

    overrides = yaml_load(checks.check_yaml(train_kwargs["cfg"])) if train_kwargs.get("cfg") else model.overrides
    custom = {
        # NOTE: handle the case when 'cfg' includes 'data'.
        "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[model.task],
        "model": model.overrides["model"],
        "task": model.task,
    }  # method defaults

    args = {**overrides, **custom, **train_kwargs, "mode": "train"}  # highest priority args on the right
    if args.get("resume"):
        args["resume"] = model.ckpt_path

    model.trainer = model._smart_load("trainer")(overrides=args, _callbacks=None)
    # model.trainer = model._smart_load("trainer")(overrides=args, _callbacks=model.callbacks)
    if not args.get("resume"):  # manually set model only if not resuming
        model.trainer.model = model.trainer.get_model(weights=model.model if model.ckpt else None, cfg=model.model.yaml)
        model.model = model.trainer.model
    model.trainer.hub_session = model.session  # attach optional HUB session
    # model.trainer.train()
    trainer = model.trainer

    world_size = 1

    trainer._setup_train(world_size)

    nb = len(trainer.train_loader)  # number of batches
    nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1
    last_opt_step = -1
    trainer.epoch_time = None
    trainer.epoch_time_start = time.time()
    trainer.train_time_start = time.time()
    if trainer.args.close_mosaic:
        base_idx = (trainer.epochs - trainer.args.close_mosaic) * nb
        trainer.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = trainer.start_epoch
    trainer.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start

    trainer.epoch = epoch
    # trainer.run_callbacks("on_train_epoch_start")
    trainer.scheduler.step()

    trainer.model.train()
    if RANK != -1:
        trainer.train_loader.sampler.set_epoch(epoch)
    pbar = enumerate(trainer.train_loader)
    if epoch == (trainer.epochs - trainer.args.close_mosaic):
        trainer._close_dataloader_mosaic()
        trainer.train_loader.reset()

    return model, trainer

def kld_train(model, train_loader=None, num_epochs=1, teacher_model=None,):
    print("KLD train")
    lr = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    train_loss = 0.0
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for idx, data in enumerate(train_loader):
            data["img"] = data["img"].to(torch.device("cuda"), non_blocking=True).float() / 255

            teacher_pred = torch.stack(teacher_model._predict_once(data['img'], embed=[9]), 0)
            pred = torch.stack(model._predict_once(data['img'], embed=[9]), 0)
            distill_loss = torch.norm((teacher_pred - pred))
            loss = distill_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_loss += epoch_loss
    print("KLD loss", train_loss)
    return train_loss

def train(model, trainer=None, train_loader=None, num_epochs=1, rnd=None, teacher_model=None,):
    lrf = 0.01
    lr = max(1 - rnd / 10, 0.1) * lrf
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    # model = model.cuda()
    train_loss = 0.0
    model.train()

    grad, no_grad = 0, 0
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for idx, data in enumerate(train_loader):
            data = trainer.preprocess_batch(data)

            loss, loss_items = model(data)
            # if teacher_model is not None:
            #     teacher_pred = torch.stack(teacher_model._predict_once(data['img'], embed=[9]), 0)
            #     pred = torch.stack(model._predict_once(data['img'], embed=[9]), 0)
            #     distill_loss = torch.norm((teacher_pred - pred))
            #     loss += distill_loss * 5

            trainer.loss, trainer.loss_items = loss, loss_items

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_loss += epoch_loss

    return train_loss

# borrowed from Pytorch quickstart example
def test(model, trainer=None, validator=None, testloader=None, device: str=None, cid=0):
    """Validate the network on the entire test set."""

    validator.device = trainer.device
    validator.data = trainer.data
    validator.loss = None
    # validator.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
    model.eval()
    validator.run_callbacks("on_val_start")
    dt = (
        Profile(device=trainer.device),
        Profile(device=trainer.device),
        Profile(device=trainer.device),
        Profile(device=trainer.device),
    )

    validator.init_metrics(de_parallel(model))
    validator.jdict = []  # empty before each val
    # pbar = TQDM(enumerate(testloader), total=len(testloader))
    with torch.no_grad():
        for batch_i, batch in enumerate(testloader):
            validator.run_callbacks("on_val_batch_start")
            validator.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = validator.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=False)

            # Loss
            with dt[2]:
                loss = model.loss(batch, preds)[1]
                if validator.loss is None:
                    validator.loss = torch.zeros_like(loss, device=device)
                validator.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = validator.postprocess(preds)

            validator.update_metrics(preds, batch)


            validator.run_callbacks("on_val_batch_end")

    stats = validator.get_stats()
    validator.check_stats(stats)
    validator.finalize_metrics()
    validator.run_callbacks("on_val_end")
    model.float()
    # results = {**stats, **trainer.label_loss_items(validator.loss.cpu() / len(testloader.dataloader), prefix="val")}
    # return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats

    # validator(model=model, trainer=model_trainer)

    return validator.loss, validator.metrics

def load_detection_model(cfg, trainer_cfg):
    cfg_dict = yaml_model_load(cfg)
    task = "detect"
    model = DetectionModel(cfg_dict, nc=cfg_dict["nc"], verbose=False)  # build model
    overrides = {}
    overrides["model"] = cfg
    overrides["task"] = task

    # Below added to allow export from YAMLs
    model.args = trainer_cfg  # combine default and model args (prefer model args)
    model.task = task

    return model

class KLD_FedAvg(FedAvg):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[
                Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            model: DetectionModel=None,
            teacher_model: DetectionModel = None,
            teacher_train_loader = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        print("Init KLD_FedAvg")
        self.model = model
        self.teacher_model = teacher_model
        self.teacher_train_loader = teacher_train_loader

    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        average_parameter = weights_to_parameters(aggregate(weights_results))

        params_dict = zip(self.model.state_dict().keys(), parameters_to_weights(average_parameter))
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(torch.device('cuda'))
        self.teacher_model.to(torch.device('cuda'))

        kld_train(model=self.model, teacher_model=self.teacher_model, train_loader=self.teacher_train_loader)

        kld_parameter = weights_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        return kld_parameter, {}


class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str = None, model_yaml: str = None, teacher_model: DetectionModel = None, kld_in_train: bool = False, dataset_yaml: str = None):
        self.cid = cid
        self.data = dataset_yaml.replace('.yaml', f'_{cid}.yaml')
        self.fed_dir = Path(self.data)

        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        init_kwargs = {
            "data": "configs/keti_fl_dataset_0.yaml",
            # "project": 'IITP',
            "batch": batch_size,
            "device": 'cuda',
            "imgsz": 640,
        }
        self.yolo_model = YOLO(model_yaml, verbose=False)
        self.yolo_model, self.trainer = self._setup_train(self.yolo_model)
        self.model = self.yolo_model.model
        self.model.args = self.trainer.args
        self.model.task = 'detect'

        self.teacher_model = teacher_model

        custom = {"rect": True}
        val_args = {**self.yolo_model.overrides, **custom, **init_kwargs, "mode": "val"}
        self.validator = DetectionValidator(args=val_args, _callbacks=self.yolo_model.callbacks)
        self.trainloader = self.trainer.train_loader
        self.testloader = self.trainer.test_loader

        self.rnd = 0
        self.data = f'configs/keti_fl_dataset_{cid}.yaml'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.kld_in_train = kld_in_train

    def _setup_train(self, model):
        train_kwargs = {
            'data': self.data,
            'batch': batch_size,
            'epochs': 1,
            'imgsz': 640,
            'verbose': False,
            'val': False,
            'save': False,

        }

        overrides = yaml_load(checks.check_yaml(train_kwargs["cfg"])) if train_kwargs.get("cfg") else model.overrides
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[model.task],
            "model": model.overrides["model"],
            "task": model.task,
        }

        args = {**overrides, **custom, **train_kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = model.ckpt_path

        model.trainer = model._smart_load("trainer")(overrides=args, _callbacks=None)
        if not args.get("resume"):  # manually set model only if not resuming
            model.trainer.model = model.trainer.get_model(weights=model.model if model.ckpt else None,
                                                          cfg=model.model.yaml)
            model.model = model.trainer.model
        model.trainer.hub_session = model.session  # attach optional HUB session
        trainer = model.trainer

        world_size = 1

        trainer._setup_train(world_size)

        nb = len(trainer.train_loader)  # number of batches
        trainer.epoch_time = None
        trainer.epoch_time_start = time.time()
        trainer.train_time_start = time.time()
        if trainer.args.close_mosaic:
            base_idx = (trainer.epochs - trainer.args.close_mosaic) * nb
            trainer.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = trainer.start_epoch
        trainer.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start

        trainer.epoch = epoch

        return model, trainer

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"fit() on client cid={self.cid} round={self.rnd}")
        self.set_parameters(parameters)
        self.model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
        self.trainloader.reset()

        client_loss = train(self.model, trainer=self.trainer, train_loader=self.trainloader,
                                rnd=self.rnd, teacher_model=self.teacher_model if self.kld_in_train else None)
        self.rnd += 1
        return self.get_parameters(), len(self.trainloader), float(client_loss)

        # client_loss = train(self.model, rnd=config["epoch_global"], cid=self.cid)
        # return self.get_parameters(), 1, 0

    # def get_trainloader(self):
    #     return self.get_dataloader(self.trainset, batch_size=batch_size, rank=-1, mode='train')
    #
    # def get_testloader(self):
    #     return self.get_dataloader(self.testset, batch_size=batch_size, rank=-1, mode='val')

    def evaluate(self, parameters, config):
        print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)
        # self.testloader.reset()

        # send model to device
        self.model.to(self.device)

        # evaluate
        loss, metrics = test(self.model, trainer=self.trainer, validator=self.validator, testloader=self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(metrics.box.map50)}

        # return float(0), 1, {"accuracy": float(0)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(10),
        "batch_size": str(batch_size),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Parameters) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.copy(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )

    # model.model.load(state_dict, verbose=False)
    model.load_state_dict(state_dict, strict=True)

def get_eval_fn(
    eval_model=None,
    trainer=None,
    validator=None,
    night_validator=None,
    testloader=None,
    night_testloader=None,
) -> Callable[[fl.common.Parameters], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Parameters) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        global global_round
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        set_weights(eval_model, weights)
        eval_model.to(device)

        torch.save(eval_model.state_dict(), f'./IITP/DetectionModel_{model_name}_last.pt')
        # model.save(f'./IITP/yolov8n_fl_full.pt')

        loss, metrics = test(eval_model, trainer=trainer, validator=validator,  testloader=testloader)
        night_loss, night_metrics = test(eval_model, trainer=trainer, validator=night_validator,  testloader=night_testloader)

        global best_accuracy
        if metrics.box.map50 > best_accuracy:
            torch.save(eval_model.state_dict(), f'./IITP/DetectionModel_{model_name}_rnd_{global_round}_best.pt')
            # torch.save(eval_model.state_dict(), f'./IITP/DetectionModel_no_kld_{model_name}_rnd_{global_round}_best.pt')
            best_accuracy = metrics.box.map50
            print("Best accuracy:", best_accuracy)

        # wandb.log({"Validation mAP50": night_metrics.box.map50})
        global_round += 1
        write_text = f'"accuracy": {metrics.box.map50}, "night_accuracy": {night_metrics.box.map50}\n'
        with open('train_log.txt', 'a') as f1:
            f1.write(write_text)
        return night_loss, {"accuracy": metrics.box.map50, "night_accuracy": night_metrics.box.map50}

    return evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl_kld', action='store_true', help='KLD after FL process')
    parser.add_argument('--no_kld', action='store_true', help='Not using KLD')
    parser.add_argument('--large_model_path', type=str, default='./IITP/YOLO_large_night_best.pt')
    parser.add_argument('--model_yaml', type=str, default='./configs/small_detection_cfg.yaml')
    parser.add_argument('--day_dataset_yaml', type=str, default='configs/keti_fl_dataset_0.yaml')
    parser.add_argument('--night_dataset_yaml', type=str, default='configs/night_keti_dataset.yaml')

    option = parser.parse_args()
    kld_in_fl = option.fl_kld
    no_kld = option.no_kld
    large_model_path = option.large_model_path
    model_yaml = option.model_yaml

    day_dataset_yaml = option.day_dataset_yaml
    night_dataset_yaml = option.night_dataset_yaml

    # os.environ['WANDB_MODE'] = 'disabled'
    # settings.update({"wandb": False})

    # wandb.init(project='IITP')
    # wandb.run.name = 'FL_train'
    # wandb.run.save()

    kwargs = {
        "data": day_dataset_yaml,
        "batch": batch_size,
        "device": 'cuda',
        "imgsz": 640,
    }

    night_kwargs = {
        "data": night_dataset_yaml,
        "batch": batch_size,
        "device": 'cuda',
        "imgsz": 640,
    }

    with open('args.yaml', 'r') as file:
        data = yaml.safe_load(file)
    # wandb.config.update(data)

    pool_size = 10
    fed_dir = '/root/data/balanced_yolo_data/'

    teacher_yolo_model = None
    if not no_kld:
        teacher_yolo_model = YOLO(large_model_path, verbose=False)
        teacher_yolo_model = teacher_yolo_model.model


    yolo_model = YOLO(model_yaml, verbose=False)
    yolo_model, trainer = _setup_train(yolo_model, data=day_dataset_yaml)

    custom = {"rect": True}
    val_args = {**yolo_model.overrides, **custom, **kwargs, "mode": "val"}
    validator = DetectionValidator(args=val_args, _callbacks=yolo_model.callbacks)

    night_val_args = {**yolo_model.overrides, **custom, **night_kwargs, "mode": "val"}
    night_validator = DetectionValidator(args=val_args, _callbacks=yolo_model.callbacks)

    trainloader = trainer.train_loader
    testloader = trainer.test_loader

    model = yolo_model.model
    model.args = trainer.args  # combine default and model args (prefer model args)
    model.task = 'detect'

    night_yolo_model = YOLO(model_yaml, verbose=False)
    night_yolo_model, night_trainer = _setup_train(yolo_model, data=night_dataset_yaml)
    night_testloader = night_trainer.test_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_initial_parameters():
        yolo_model = YOLO(model_yaml, verbose=False)
        yolo_model, trainer = _setup_train(yolo_model)

        model = yolo_model.model
        # model = DetectionModel(model_yaml, nc=93, verbose=False)
        # model.load_state_dict(torch.load('./IITP/DetectionModel_no_kld_small_detection_cfg_rnd_12_best.pt'))

        model.args = trainer.args  # combine default and model args (prefer model args)
        model.task = 'detect'

        weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return weights

    # configure the strategy
    initial_parameters: Parameters = get_initial_parameters()
    print("KLD_FedAvg usage", kld_in_fl)

    if kld_in_fl:  # KLD in FL
        strategy = KLD_FedAvg(
            fraction_fit=0.1,
            min_fit_clients=max(int(0.5 * pool_size), 1),
            min_available_clients=pool_size,  # All clients should be available
            on_fit_config_fn=fit_config,
            eval_fn=get_eval_fn(eval_model=model, trainer=trainer, validator=validator, night_validator=night_validator,
                                testloader=testloader, night_testloader=night_testloader),
            initial_parameters=get_initial_parameters(),
            model=model,
            teacher_model=teacher_yolo_model,
            teacher_train_loader=trainloader,
        )

    else: # KLD in local or No KLD
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            min_fit_clients=max(int(0.5 * pool_size), 1),
            min_available_clients=pool_size,  # All clients should be available
            on_fit_config_fn=fit_config,
            eval_fn=get_eval_fn(eval_model=model, trainer=trainer, validator=validator, night_validator=night_validator, testloader=testloader, night_testloader=night_testloader),  # centralised testset evaluation of global model
            initial_parameters=get_initial_parameters(),
         )


    strategy.initial_parameters = initial_parameters


    def client_fn(cid: str):
        client = CifarRayClient(cid, model_yaml=model_yaml, teacher_model=teacher_yolo_model, kld_in_train=False if no_kld else kld_in_fl, dataset_yaml=day_dataset_yaml)
        return client

    # (optional) specify ray config
    ray_config = {
        "include_dashboard": False,
        "num_gpus": 1,
        "num_cpus": 1,
    }

    RANK = -1

    f = open('train_log.txt', 'w')
    f.close()

    # start simulation
    output = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        num_rounds=50,
        strategy=strategy,
        ray_init_args=ray_config,
    )

    import json

    with open ('fl_yolo.json', 'w') as f:
        json.dump(output.metrics_centralized, f)

