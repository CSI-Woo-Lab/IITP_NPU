# IITP_NPU Federated Learning Framework Based on Flower Framework

### This project was supported in part by the Institute for Information and Communications Technology Planning and Evaluation (IITP) under Grant 2021-0-00875

## Introduction
#### Support CUDA Version
* #### 12.x
#### Federated Learning Evaluation 
* #### Dataet: CIFAR-10
* #### Client \#: 100
* #### Evaluation Metric: Classification

## System Architecture 
![architecture](/asset/architecture.bmp)

## Dataset: CIFAR-10
![CIFAR-10 Dataset Examples](/asset/cifar.png)

## Our Framework Conformance Verification
![CIFAR-10 Dataset Examples](/asset/verification.png)


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

### Excute (for CPU)
``` bash
python main.py --device=cpu --num=0
```

### Excute (for GPU)
``` bash
python main.py --device=gpu --num=0
```

### Evaluation Results
``` bash
```
