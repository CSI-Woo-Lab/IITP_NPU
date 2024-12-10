#!/bin/bash

# runing rate와 momentum 값의 배열 정의
learning_rates=(0.01 0.05 0.15 0.25)
momentums=(0.9)

# 각 learning rate와 momentum 조합에 대해 main.py 실행
for lr in "${learning_rates[@]}"; do
    for momentum in "${momentums[@]}"; do
        echo "Running main.py with lr=$lr and momentum=$momentum"
        python3 main.py --lr $lr --momentum $momentum
    done
done
