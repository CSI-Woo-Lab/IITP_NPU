# How to use FL training framework

## 1. Dataset 구조

YOLO 데이터셋 구조를 따르며, Federated Learning(FL)을 위해 데이터셋을 여러 개의 디렉토리로 나누어야 합니다. 해당 디렉토리 경로가 아래 config 파일 내에 train 및 val에 적혀 있어야 합니다. 아래는 그 예시입니다:

```
└──0
   ├── images
   │   ├── train
   │   └── val
   └── labels
       ├── train
       └── val
└──1
   ├── images
   │   ├── train
   │   └── val
   └── labels
       ├── train
       └── val
└──2
...
```
## 2. Config 파일 구조

config 파일은 아래와 같은 구조를 가집니다:

- **class names**: index에 해당하는 class 이름 (더미 값 넣어도 무관)
- **nc**: class 개수
- **train**: train에 사용할 이미지 경로
- **val**: test에 사용할 이미지 경로
- label은 train과 동일한 디렉토리에서 label 경로를 사용

또한, fl 구조로 변형하다보니, 분리된 디렉토리에 각각 해당하는 config 파일들이 필요합니다. (configs 파일 참고, 10 client로 실험 했음)

```yaml
# 예시 config 파일 구조
names:
  0: class0
  1: class1
nc: 2
train: path/to/train/images
val: path/to/val/images
```


## 3. Training
```
python main.py --configs ~
```

## 4. Testing
