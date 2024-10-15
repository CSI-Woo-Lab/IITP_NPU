import os
import shutil
import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import yaml
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError

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

# sample_images 디렉토리 생성 (존재하지 않는 경우)
def create_yolo_dataset(image_dir, label_dir, output_dir, val_ratio=0.05):
    """
    Create a dataset for YOLO model training.

    Parameters:
    - image_dir (str): Path to the directory containing all images.
    - label_dir (str): Path to the directory containing all labels.
    - output_dir (str): Path to the output directory for YOLO dataset.
    - val_ratio (float): Ratio of validation data (default is 0.2).
    """
    # Create output directories
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    train_label_dir = os.path.join(output_dir, 'labels', 'train')
    val_label_dir = os.path.join(output_dir, 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get all image and label files
    image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    print(len(image_files), len(label_files))

    # Check if image and label files match
    if len(image_files) != len(label_files):
        raise ValueError("Number of images and labels do not match.")

    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=val_ratio, random_state=42
    )

    # Copy files to the output directories
    for img, lbl in zip(tqdm.tqdm(train_images), train_labels):
        shutil.copy(os.path.join(image_dir, img), train_img_dir)
        shutil.copy(os.path.join(label_dir, lbl), train_label_dir)

    for img, lbl in zip(tqdm.tqdm(val_images), val_labels):
        shutil.copy(os.path.join(image_dir, img), val_img_dir)
        shutil.copy(os.path.join(label_dir, lbl), val_label_dir)

    print(f"Dataset created successfully. Train set: {len(train_images)}, Validation set: {len(val_images)}")


def sampling_dataset(source, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)

    # 이미지 파일 목록 가져오기 및 이름 순으로 정렬
    files = sorted(os.listdir(source))

    # 앞에서 100개의 파일만 선택
    selected_files = files[::10]

    # 파일 복사
    for file_name in selected_files:
        source_path = os.path.join(source, file_name)
        destination_path = os.path.join(dest, file_name)
        shutil.copy2(source_path, destination_path)

    print(f"Selected and copied {len(selected_files)} files to '{dest}' directory.")


def plot_training_accuracies(file_name):
    colors = ['#AEC6CF', '#FFB347', '#B39EB5', '#FF6961', '#77DD77', '#F49AC2', '#CFCFC4', '#FFD1DC', '#779ECB',
              '#966FD6']


    plt.figure()

    if file_name.endswith(".json"):
        filepath = file_name
        with open(filepath, 'r') as file:
            data = json.load(file)
            epochs, accuracies = zip(*data["accuracy"])
            # Filter data to plot every 10 epochs
            filtered_epochs = [epoch for epoch in epochs]
            if filtered_epochs[0] < 0:
                for idx, epoch in enumerate(filtered_epochs):
                    filtered_epochs[idx] = epoch - epochs[0]
            filtered_accuracies = [accuracies[i] for i in filtered_epochs]
            plt.plot(filtered_epochs, filtered_accuracies, marker='o', linewidth=2, markersize=4, color=colors[0],
                     label=os.path.splitext(file_name)[0])

    no_kld_lod = [
        0.10,  # 0
        0.419,
        0.559,
        0.730,
        0.738,
        0.835,
        0.859,
        0.869,
        0.893,
        0.903,
        0.905,
        0.918,  # 11
        0.927,
        0.923,
        0.923,
        0.930,
        0.932,
        0.937,
        0.934,
        0.938,
        0.940,
        0.941,
        0.939,
        0.942,
        0.945,
        0.942,
        0.948,
        0.947,
        0.950,
        0.948,
        0.951,
        0.948,
        0.946,
        0.947,
        0.943,
        0.943,
        0.944,
        0.942,
        0.948,
        0.947,
        0.950,
        0.948,
        0.951,
        0.950,
        0.947,
        0.946,
        0.954,
        0.948,
        0.946,
        0.946
    ]

    plt.plot(filtered_epochs, no_kld_lod, marker='o', linewidth=2, markersize=4, color=colors[1],
             label="fl_no_kld_log")

    plt.title('Training Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0.5, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(f'total_accuracy.png')


def plot_all_training_accuracies(directory):
    colors = ['#AEC6CF', '#FFB347', '#B39EB5', '#FF6961', '#77DD77', '#F49AC2', '#CFCFC4', '#FFD1DC', '#779ECB',
              '#966FD6']


    plt.figure()

    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                epochs, accuracies = zip(*data["accuracy"])
                # Filter data to plot every 10 epochs
                filtered_epochs = [epoch for epoch in epochs]
                filtered_accuracies = [accuracies[i] for i in filtered_epochs]
                plt.plot(filtered_epochs, filtered_accuracies, marker='o', linewidth=2, markersize=4, color=colors[i % len(colors)],
                         label=os.path.splitext(filename)[0])


    plt.title('Training Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0.5, 1.0)
    plt.legend(loc='lower right')
    plt.savefig(f'total_accuracy.png')


def plot_class_num_and_map50(excel_file):
    # 엑셀 파일에서 데이터 읽기
    df = pd.read_excel(excel_file)

    # 이미지 수 기준으로 데이터 정렬
    df_sorted = df.sort_values(by='Images', ascending=False)

    # 그래프 생성
    fig, ax1 = plt.subplots()

    # x축 설정
    x = df_sorted['Labels']

    # 첫 번째 y축: Images
    ax1.set_xlabel('Labels')
    ax1.set_ylabel('Images')
    ax1.plot(x, df_sorted['Images'], color='tab:blue', label='Images')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, df_sorted['Images'].max() + 30)  # 최소값을 0으로 설정

    # x축 라벨 기울이기
    plt.xticks(rotation=70)

    # 두 번째 y축: mAP50
    ax2 = ax1.twinx()
    ax2.set_ylabel('mAP50')
    ax2.plot(x, df_sorted['mAP50'], color='tab:red', label='mAP50')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0.2, 1.0)  # 범위를 0.2 ~ 1.0으로 설정

    # 제목 설정
    plt.title('Frequency and mAP50')
    fig.tight_layout()

    # 범례 설정
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.savefig(f'class_num_and_mAP50.png')

def result_to_excel(data_path: str):
    data = pd.read_csv(os.path.join('trashcan', data_path), delim_whitespace=True)
    data.to_excel('test_result.xlsx', index=False)


def calculate_map_averages(excel_file):
    # 엑셀 파일에서 데이터 읽기
    df = pd.read_excel(excel_file)

    # 이미지 수 기준으로 데이터 정렬
    df_sorted = df.sort_values(by='Images', ascending=False)

    # 상위 30%와 하위 30%의 인덱스 계산
    num_rows = len(df)
    top_30_percent = int(num_rows * 0.3)
    bottom_30_percent = num_rows - top_30_percent

    # 상위 30%와 하위 30% 데이터 선택
    top_30_data = df_sorted.head(top_30_percent)
    bottom_30_data = df_sorted.tail(top_30_percent)

    # mAP50 및 mAP50-95의 평균 계산
    top_30_map50_avg = top_30_data['mAP50'].mean()
    top_30_map50_95_avg = top_30_data['mAP50-95'].mean()

    bottom_30_map50_avg = bottom_30_data['mAP50'].mean()
    bottom_30_map50_95_avg = bottom_30_data['mAP50-95'].mean()

    top_30_images_sum = top_30_data['Images'].sum()
    bottom_30_images_sum = bottom_30_data['Images'].sum()

    print(f"Top 30% mAP50 Average: {top_30_map50_avg}")
    print(f"Top 30% mAP50-95 Average: {top_30_map50_95_avg}")
    print(f"Top 30% Images Sum: {top_30_images_sum}")
    print(f"Bottom 30% mAP50 Average: {bottom_30_map50_avg}")
    print(f"Bottom 30% mAP50-95 Average: {bottom_30_map50_95_avg}")
    print(f"Bottom 30% Images Sum: {bottom_30_images_sum}")

    return (top_30_map50_avg, top_30_map50_95_avg, bottom_30_map50_avg, bottom_30_map50_95_avg)


#############################################
######## Make FL dataset stucture ###########
#############################################

def resize_image(image, target_height=640):
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = image.resize((new_width, target_height))
    return resized_image


def save_pt_file(images, labels, file_path):
    data = [images, labels]
    torch.save(data, file_path)


def create_federated_yolov8_dataset(img_src_dir, label_src_dir, dst_dir, train_ratio=0.97):
    image_files = sorted([f for f in os.listdir(img_src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(label_src_dir) if f.endswith('.txt')])

    print(len(image_files), len(label_files))
    # 이미지 파일과 레이블 파일의 짝을 맞춤
    pairs = list(zip(image_files, label_files))

    # train과 val 데이터셋으로 분할
    train_pairs, val_pairs = train_test_split(pairs, train_size=train_ratio, random_state=42)

    num_splits = 10

    # train과 val 데이터를 각각 num_splits개로 나눔
    train_splits = [train_pairs[i::num_splits] for i in range(num_splits)]
    val_splits = [val_pairs[i::num_splits] for i in range(num_splits)]

    # 무작위로 섞기
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)


    for i in range(num_splits):
        split_image_train_dir = os.path.join(dst_dir, f'{i}', 'images', 'train')
        split_label_train_dir = os.path.join(dst_dir, f'{i}', 'labels', 'train')
        split_image_val_dir = os.path.join(dst_dir, f'{i}', 'images', 'val')
        split_label_val_dir = os.path.join(dst_dir, f'{i}', 'labels', 'val')

        os.makedirs(split_image_train_dir, exist_ok=True)
        os.makedirs(split_label_train_dir, exist_ok=True)
        os.makedirs(split_image_val_dir, exist_ok=True)
        os.makedirs(split_label_val_dir, exist_ok=True)

        for image_file, label_file in tqdm.tqdm(train_splits[i]):
            shutil.copy(os.path.join(img_src_dir, image_file), os.path.join(split_image_train_dir, image_file))
            shutil.copy(os.path.join(label_src_dir, label_file), os.path.join(split_label_train_dir, label_file))

        for image_file, label_file in tqdm.tqdm(val_splits[i]):
            shutil.copy(os.path.join(img_src_dir, image_file), os.path.join(split_image_val_dir, image_file))
            shutil.copy(os.path.join(label_src_dir, label_file), os.path.join(split_label_val_dir, label_file))


def create_subset_dataset(src_image_dir, src_label_dir, dst_image_dir, dst_label_dir, subset_ratio=0.3):
    # 소스 디렉토리에서 이미지 및 레이블 파일 목록 가져오기
    image_files = sorted([f for f in os.listdir(src_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(src_label_dir) if f.endswith('.txt')])

    # 이미지 파일과 레이블 파일의 짝을 맞춤
    pairs = list(zip(image_files, label_files))

    # 전체 파일 목록에서 subset_ratio 비율만큼 무작위로 선택
    subset_size = int(len(pairs) * subset_ratio)
    subset_pairs = random.sample(pairs, subset_size)

    # 대상 디렉토리 생성
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    # 선택된 파일들을 대상 디렉토리로 복사
    for image_file, label_file in tqdm.tqdm(subset_pairs):
        shutil.copy(os.path.join(src_image_dir, image_file), os.path.join(dst_image_dir, image_file))
        shutil.copy(os.path.join(src_label_dir, label_file), os.path.join(dst_label_dir, label_file))


def modify_and_save_yaml(original_yaml, dst_dir):
    with open(f'{original_yaml}.yaml', 'r') as file:
        original_data = yaml.safe_load(file)

    for i in range(10):
        modified_data = original_data.copy()
        modified_data['train'] = str(modified_data['train'].replace('fl_yolo_data', f'fl_yolo_data/{i}'))
        modified_data['val'] = str(modified_data['val'].replace('fl_yolo_data', f'fl_yolo_data/{i}'))

        new_yaml_file = os.path.join(dst_dir, f'{original_yaml}_{i}.yaml')
        with open(new_yaml_file, 'w') as yaml_file:
            yaml.dump(modified_data, yaml_file, default_flow_style=False)
        print(f"Created YAML file: {new_yaml_file}")


#############################################


# 디렉토리 경로 설정
source_img_dir = '/root/data/images'
destination_img_dir = '/root/data/balanced_yolo_data/images'

source_label_dir = '/root/data/labels'
destination_label_dir = '/root/data/balanced_yolo_data/labels'

# result_to_excel('yolo_norm_dataset_val.txt')
# calculate_map_averages(os.path.join('results', 'norm_data_test_result.xlsx'))

plot_training_accuracies('fl_kld_log.json')
# plot_all_training_accuracies('results')
# plot_class_num_and_map50(os.path.join('results', 'norm_data_test_result.xlsx'))

# process_and_adjust_labels(source_img_dir, source_label_dir, destination_img_dir, destination_label_dir, class_names)

# collect_and_plot_cumulative_labels(source_label_dir)

# sampling_dataset(source_img_dir, destination_img_dir)
# sampling_dataset(source_label_dir, destination_label_dir)

# modify_and_save_yaml('keti_fl_dataset', 'configs/')

balance_image_dir = '/root/data/balanced_yolo_data/images/train'
balance_label_dir = '/root/data/balanced_yolo_data/labels/train'
fl_balance_image_dir = '/root/data/fl_yolo_data/images/train'
fl_balance_label_dir = '/root/data/fl_yolo_data/labels/train'
balance_data_dir = '/root/data/balanced_yolo_data/'
fl_data_dir = '/root/data/fl_yolo_data'

# create_yolo_dataset(destination_img_dir, destination_label_dir, '/root/data/new_balanced_yolo_data')
# create_federated_yolov8_dataset(destination_img_dir, destination_label_dir, fl_data_dir)
# create_subset_dataset(balance_image_dir, balance_label_dir, small_balance_image_dir, small_balance_label_dir)

def sample_yolo_dataset(src_dir, dst_dir, sample_ratio=0.1):
    def sample_and_copy(src_subdir, dst_subdir):
        # 원본 디렉토리에서 파일 목록 가져오기
        files = sorted(os.listdir(src_subdir))

        # 샘플링할 파일 개수 계산
        sample_size = int(len(files) * sample_ratio)

        # 파일 무작위로 선택
        sampled_files = random.sample(files, sample_size)

        # 출력 디렉토리 생성
        os.makedirs(dst_subdir, exist_ok=True)

        # 파일 복사
        for file_name in sampled_files:
            src_file_path = os.path.join(src_subdir, file_name)
            dst_file_path = os.path.join(dst_subdir, file_name)
            shutil.copy(src_file_path, dst_file_path)

    # 각 서브 디렉토리에 대해 샘플링 및 복사
    subdirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for subdir in subdirs:
        src_subdir = os.path.join(src_dir, subdir)
        dst_subdir = os.path.join(dst_dir, subdir)
        if os.path.exists(src_subdir):
            sample_and_copy(src_subdir, dst_subdir)
        else:
            print(f"Warning: {src_subdir} does not exist and will be skipped.")

sample_yolo_dataset('/root/data/night_fl_yolo_data/', '/root/data/night_fl_yolo_data/')