# IITP_NPU Federated Framework Based on Flower Framework for CIFAR-10 Federated Learning Evaluation

### This project was supported in part by the Institute for Information and Communications Technology Planning and Evaluation (IITP) under Grant 2021-0-00875

## Introduction
#### Federated Learning Evaluation 
* #### Dataet: CIFAR-10
* #### Client \#: 100
* #### Evaluation Metric: Classification

## System Architecture 
![architecture](/asset/architecture.png)

## Dataset: CIFAR-10
![CIFAR-10 Dataset Examples](/asset/cifar.png)

## Install

### Docker Pull & Run
``` bash
docker pull mkris0714/iitp_npu:latest
docker run --gpus all -e LC_ALL=C.UTF-8 -p 8080:8080 -it mkris0714/iitp_npu:latest /bin/bash
```

### Git Clone
``` bash
git clone https://github.com/CSI-Woo-Lab/IITP_NPU.git
```

### Excute
``` bash
cd ~/2023
python main.py
```

### Evaluation Results Checking
``` bash
```
