import cv2
import numpy as np
import os
from PIL import Image, UnidentifiedImageError


def process_image(image_path, output_path):
    # 이미지 로드
    input_image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # 밝은 부분의 밝기 수준 낮추기
    h, s, v = cv2.split(hsv_image)  # HSV 기반으로 분할
    bright_mask = v > 180
    v[bright_mask] = v[bright_mask] * 0.6
    dark_hsv_image = cv2.merge([h, s, v])
    dark_image = cv2.cvtColor(dark_hsv_image, cv2.COLOR_HSV2BGR)

    # 밝기 및 명도 조정
    alpha = 0.8  # 명도 (0.0 - 1.0)
    beta = -25  # 밝기 (-100 - 100)
    dark_image = cv2.convertScaleAbs(dark_image, alpha=alpha, beta=beta)

    # 파란색 레이어 추가
    blue_layer = np.zeros_like(dark_image)
    blue_layer[:, :, 0] = 50

    # 이미지 병합
    night_image = cv2.addWeighted(dark_image, 1.0, blue_layer, 0.5, 0)

    # 이미지 저장
    cv2.imwrite(output_path, night_image)


def process_directory(input_dir, output_dir):
    # 입력 디렉토리 내의 모든 하위 디렉토리를 순회
    for root, dirs, files in os.walk(input_dir):
        # 모든 파일을 처리
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자 체크
                try:
                    # 이미지 로드
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)

                    # 출력 디렉토리 생성
                    os.makedirs(output_subdir, exist_ok=True)

                    input_file_path = os.path.join(root, file)
                    output_file_path = os.path.join(output_subdir, file)

                    # 이미지 처리 및 저장
                    process_image(input_file_path, output_file_path)

                    print(f"Processed and saved: {output_file_path}")
                except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
                    print(f"Skipping {e}")



# 예시 사용법
input_directory = '/root/data/fl_yolo_data'
output_directory = '/root/data/night_fl_yolo_data'

process_directory(input_directory, output_directory)