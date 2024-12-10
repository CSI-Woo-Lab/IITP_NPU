import os
from PIL import Image
import argparse
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load, torch_safe_load, DetectionModel
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
    RANK,
)

class_names = [
    'Cross Intersection',
    'T-Intersection ',
    'Use both lane',
    'Intersection 1',
    'Intersection 2',
    'Seperation for direction',
    'Right-Merge',
    'U-Turn',
    'Roundabout',
    'Left Turn and Right Turn',
    'Right-Curve',
    'Left-Curve',
    'Left Turn and U-Turn',
    'Left-Double reverse-Curve',
    'Keep Left',
    'Up-Hill',
    'Down-Hill',
    'Keep Right',
    'Right Lane Ends',
    'Left Lane Ends',
    'Straight Thru and Left Turn',
    'Use both lane',
    'Divided Road',
    'Straight Thru and Right Turn',
    'Signal',
    'Slippery when wet',
    'Riverside Road',
    'Rough Road',
    'Speed Bump',
    'Keepout Rocks',
    'Cross Walk',
    'Watchout Children',
    'Watchout bicycle',
    'Road Work',
    'Bus Lane',
    'Side Wind',
    'Tunnel',
    'Bridge',
    'Caution Wild Life',
    'Danger',
    'Left Turn',
    'Right Turn',
    'Left turn signal yield on Green',
    'No Trucks Allowed',
    'One way(3)',
    'No motorcycles Allowed',
    'No Automobiles Allowed',
    'Straight Thru',
    'No bicycles Allowed',
    'No Entry',
    'No Straight Thru',
    'No Right Turn',
    'No Left Turn',
    'No U-Turn',
    'Do Not Pass',
    'No Parking or Standing',
    'No Parking',
    'Gap between cars',
    'Speed Limit 30',
    'Speed Limit 40',
    'Speed Limit 50',
    'Speed Limit 60',
    'Speed Limit 70',
    'Speed Limit 80',
    'Speed Limit 90',
    'Speed Limit 100',
    'Speed Limit 110',
    'Roundabout',
    'Bicycle Pedestrian Detour',
    'Speed Limit Minimum 50',
    'One way(1)',
    'Bicycle Cross Walk',
    'Caution Children',
    'Bicycle Only',
    'Cross Walk',
    'Parking Lot',
    'Slow Down',
    'Stop',
    'Yield',
    'No Pedestrian Passing',
    'Seperation Bicycle and Pedestrian',
    'Driveway',
    'VMS',
    'road usable',
    'road unusable',
    'unknown',
    'red',
    'yellow',
    'green',
    'left',
    'green_left'
]


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False, raw_model=None):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    param_model = ckpt.get("ema")
    param = OrderedDict({name: torch.from_numpy(np.copy(param)) for name, param in param_model.state_dict().items()})
    model = raw_model
    model.load_state_dict(param, strict=True)
    # model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt

def print_results(val):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(val.metrics.keys)  # print format
        print(pf % ("all", val.seen, val.nt_per_class.sum(), *val.metrics.mean_results()))

        # Print results per class
        for i, c in enumerate(val.metrics.ap_class_index):
            print(
                pf % (class_names[int(val.names[c])], val.nt_per_image[c], val.nt_per_class[c], *val.metrics.class_result(i))
            )


def calculate_map50_averages(class_counts, map50_values):
    class_map_pairs = list(zip(class_counts, map50_values))

    class_map_pairs_sorted = sorted(class_map_pairs, key=lambda x: x[0], reverse=True)

    num_classes = len(class_map_pairs_sorted)
    top_30_percent_count = int(num_classes * 0.3)
    bottom_30_percent_count = top_30_percent_count

    top_30_data = class_map_pairs_sorted[:top_30_percent_count]
    bottom_30_data = class_map_pairs_sorted[-bottom_30_percent_count:]

    top_30_map50_avg = sum([x[0] * x[1] for x in top_30_data]) / sum([x[0] for x in top_30_data])
    bottom_30_map50_avg = sum([x[0] * x[1] for x in bottom_30_data]) / sum([x[0] for x in bottom_30_data])

    return top_30_map50_avg, bottom_30_map50_avg

def custom_val(
        model,
        validator=None,
        **kwargs,
):
    custom = {"rect": True}  # method defaults
    args = {**model.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right
    validator = DetectionValidator(args=args, _callbacks=model.callbacks)
    validator(model=model.model)
    model.metrics = validator.metrics
    return validator

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='KLD_model_best.pt')
    parser.add_argument('--model_yaml', type=str, default='configs/small_detection_cfg.yaml')
    parser.add_argument('--dataset_yaml', type=str, default='configs/keti_fl_dataset.yaml')
    parser.add_argument('--save_dir', type=str, default='test')

    option = parser.parse_args()

    kld_in_fl = option.fl_kld
    no_kld = option.no_kld
    model_path = option.model_path
    model_yaml = option.model_yaml
    dataset_yaml = option.dataset_yaml
    save_dir = option.save_dir

    model = YOLO(model_yaml, verbose=True)


    detection_model = DetectionModel(model_yaml, nc=93)
    detection_model.load_state_dict(torch.load(model_path))
    model.model = detection_model

    kwargs = {
        "data": dataset_yaml,
        "batch": 4,
        "device": 'cuda',
        "imgsz": 640,
        "project": 'save_dir',
    }

    val = custom_val(model=model, **kwargs)
    map50 = []
    idxs = []
    cls_freq = val.nt_per_image
    map50_res = val.metrics.class_result
    with open('class_map', 'w') as f:
        f.write(f'[\n')
        for i, c in enumerate(val.metrics.ap_class_index):
            idxs.append(class_names[c])
            map50.append(map50_res(i)[2])
            f.write(f'    [\'{class_names[c]}\', {str(map50_res(i)[2])}],\n')
        f.write(f']')

    new_freq = []
    for freq in cls_freq:
        if freq != 0:
            new_freq.append(freq)

    print(np.max(map50), np.min(map50))  # max / min mAP
    print(calculate_map50_averages(new_freq, map50))  # top 30% and bottom 30% mAP average
