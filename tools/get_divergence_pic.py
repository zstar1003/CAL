'''
选择委员会分歧度最大的n张图片
'''
import os
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
from utils.augmentations import letterbox, hist_equalize
from utils.general import non_max_suppression
from models.common import DetectMultiBackend

# yolo prams
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # Convert back to BGR
    return im


def calculate_inconsistency(*confidences_lists):
    """Calculate inconsistency as the variance of confidence scores across different augmentations"""
    all_confidences = np.concatenate(confidences_lists)
    return np.var(all_confidences)

def random_gaussian_blur(im):
    return cv2.GaussianBlur(im, (5, 5), 0)

def random_median_blur(im):
    return cv2.medianBlur(im, 5)

# 在图像处理流程中加入选择的数据增强方法
'''
HSV颜色空间调整 (augment_hsv)：
直方图均衡化 (hist_equalize)：可以改善图像的对比度，适用于图像太暗或太亮的情况。
高斯模糊 (cv2.GaussianBlur)：对图像应用轻微的模糊效果，可以帮助模型对抗噪声。
中值滤波 (cv2.medianBlur)：可以减少图像噪声，同时保持边缘信息。
'''
def augment_images(im):
    imgs = [im]
    imgs.append(augment_hsv(im))  # HSV颜色空间调整
    imgs.append(hist_equalize(im, clahe=True, bgr=True))  # 直方图均衡化
    imgs.append(random_gaussian_blur(im))  # 高斯模糊
    imgs.append(random_median_blur(im))  # 中值滤波
    return imgs


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder, copy_images_folder, copy_labels_folder, n=100):
    # Read all files from source folders
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))

    # 确保目标文件夹存在
    os.makedirs(copy_images_folder, exist_ok=True)
    os.makedirs(copy_labels_folder, exist_ok=True)

    # Read all files from target folders (to ensure we don't select these)
    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))

    # Filter out files that already exist in the target folders
    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels
    # Ensure the filenames without extension match between images and labels
    available_files = set(file.split('.')[0] for file in available_images) & set(file.split('.')[0] for file in available_labels)
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = "../weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/VisDrone.yaml', fp16=False)

    # Compute confidences for each file
    file_inconsistencies = []
    for file in tqdm(available_files):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img0 = cv2.imread(img_path)
        imgs = augment_images(img0)
        confidences = []
        for img in imgs:
            img = letterbox(img, 1280, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            im = im.unsqueeze(0)
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            if pred is not None and len(pred):
                confidences.append(pred[:, 4].cpu().numpy())
            else:
                confidences.append(np.array([]))

        file_inconsistency = calculate_inconsistency(*confidences)
        file_inconsistencies.append((file, file_inconsistency))

    # Select files with the highest inconsistency
    selected_files = [file for file, _ in sorted(file_inconsistencies, key=lambda x: -x[1])[:n]]
    # Copy the selected images and labels to the target folders
    for file in tqdm(selected_files):
        image_file = next((img for img in available_images if img.startswith(file)), None)
        label_file = next((lbl for lbl in available_labels if lbl.startswith(file)), None)

        if image_file and label_file:
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
            shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))


if __name__ == '__main__':
    source_images_folder = "../dataset/VisDrone/train/images"
    source_labels_folder = "../dataset/VisDrone/train/labels"
    target_images_folder = "../dataset/VisDrone_part/init/images"
    target_labels_folder = "../dataset/VisDrone_part/init/labels"
    copy_images_folder = "../dataset/VisDrone_part/inconsistency_select/1/images"
    copy_labels_folder = "../dataset/VisDrone_part/inconsistency_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder,target_labels_folder, copy_images_folder, copy_labels_folder, n=pic_num)
