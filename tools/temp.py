import os
import shutil
import torch
import numpy as np
import cv2
import sys
from tqdm import tqdm
import random
import argparse

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的路径
root_dir = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(root_dir)

from utils.augmentations import letterbox
from utils.downloads import attempt_download
from utils.general import intersect_dicts, non_max_suppression
from models.backbone import DetectionModelBackboneFeature, DetectionModelCombined

def extract_features(model, img):
    with torch.no_grad():
        features = model(img)
    return features.squeeze()

def model_head_from_features(model, features):
    """
    从扰动后的特征中通过模型的头部获取置信度。
    """
    with torch.no_grad():
        output = model(features)  # 只获取输出，不再重新计算特征
        pred = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)[0]
    return pred

def image_flip(img, flip_code):
    """
    翻转图像（水平或垂直）。
    """
    return cv2.flip(img, flip_code)

def hsv_adjustment(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """
    调整图像的 HSV 颜色空间。
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(img_hsv)
    h = ((h + random.uniform(-1, 1) * h_gain * 180) % 180).astype(np.uint8)
    s = np.clip(s * (1 + random.uniform(-1, 1) * s_gain), 0, 255).astype(np.uint8)
    v = np.clip(v * (1 + random.uniform(-1, 1) * v_gain), 0, 255).astype(np.uint8)
    img_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def gaussian_blur(img, kernel_size=(5, 5), sigma=1.0):
    """
    对图像应用高斯模糊。
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)

def add_noise(features, noise_level=0.1):
    """
    给特征添加高斯噪声。
    """
    noise = torch.randn_like(features) * noise_level
    return features + noise

def spatial_dropout(features, p=0.2):
    mask = torch.ones_like(features)
    mask = torch.nn.functional.dropout2d(mask, p=p, training=True)
    return features * mask

def channel_wise_dropout(features, p=0.2):
    if features.dim() == 3:
        features = features.unsqueeze(0)
    batch_size, num_channels, _, _ = features.size()
    drop_mask = (torch.rand(batch_size, num_channels, 1, 1, device=features.device) > p).float()
    return features * drop_mask

def add_noise_to_confidence(confidences, noise_level=0.1):
    """
    对置信度添加高斯噪声。
    """
    noise = torch.randn_like(confidences) * noise_level
    return confidences + noise

def calculate_entropy(features):
    """
    计算特征或置信度的检测信息熵。
    """
    p = torch.softmax(features, dim=1)
    log_p = torch.log(p + 1e-10)
    entropy = -torch.sum(p * log_p, dim=1, keepdim=True)
    return entropy

def load_labels(label_path):
    """
    加载标签文件。
    """
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.split())) for line in f.readlines()]
    return labels

def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model_backbone, model_combined, device, n=100, perturbation_methods=None):
    if perturbation_methods is None:
        perturbation_methods = []

    # 读取源文件夹中的所有文件
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))

    # 确保目标文件夹存在
    os.makedirs(copy_images_folder, exist_ok=True)
    os.makedirs(copy_labels_folder, exist_ok=True)

    # 读取目标文件夹中的所有文件（确保不选择这些文件）
    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))

    # 过滤出目标文件夹中不存在的文件
    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels

    # 确保图像和标签的文件名（不包括扩展名）匹配
    available_files = set(file.split('.')[0] for file in available_images) & set(
        file.split('.')[0] for file in available_labels)

    # 初始化保存信息熵差值的列表
    file_scores = []

    # 遍历目标文件夹中的图像，按顺序应用扰动并计算分数
    for file in tqdm(available_files, desc="Calculating scores"):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = np.ascontiguousarray(img)

        label_path = os.path.join(source_labels_folder, f"{file}.txt")

        perturbed_img = img.copy()

        # 图像级扰动
        if 'flip' in perturbation_methods:
            flip_code = random.choice([-1, 0, 1])
            perturbed_img = image_flip(perturbed_img, flip_code)

        if 'hsv' in perturbation_methods:
            perturbed_img = hsv_adjustment(perturbed_img)

        if 'blur' in perturbation_methods:
            perturbed_img = gaussian_blur(perturbed_img)

        # 转换图像用于模型输入
        im_tensor = torch.from_numpy(perturbed_img.transpose((2, 0, 1))[::-1]).to(device).float().unsqueeze(0) / 255.0

        # 特征级扰动
        features = extract_features(model_backbone, im_tensor)
        perturbed_features = features.clone()

        if 'noise' in perturbation_methods:
            perturbed_features = add_noise(perturbed_features)

        if 'spatial_dropout' in perturbation_methods:
            perturbed_features = spatial_dropout(perturbed_features)

        if 'channel_wise_dropout' in perturbation_methods:
            perturbed_features = channel_wise_dropout(perturbed_features)

        # 初始化组合模型，并传入扰动后的特征
        combined_model = DetectionModelCombined(cfg=args.data, backbone_features=perturbed_features).to(device)
        combined_model.load_state_dict(model_backbone.state_dict(), strict=False)  # 加载相同的权重

        # 使用扰动后的特征获取输出（即置信度等）
        if 'conf_noise' in perturbation_methods:
            pred = model_head_from_features(combined_model, perturbed_features)
            if pred is not None and len(pred):
                confidences = pred[:, 4]  # 获取原始置信度
                perturbed_confidences = add_noise_to_confidence(confidences, noise_level=0.1)
                original_conf_entropy = calculate_entropy(confidences.unsqueeze(0))  # 计算原始置信度的信息熵
                perturbed_conf_entropy = calculate_entropy(perturbed_confidences.unsqueeze(0))  # 计算扰动后置信度的信息熵
                entropy_diff = torch.abs(perturbed_conf_entropy - original_conf_entropy).mean().item()
                file_scores.append((img_name, entropy_diff))
        else:
            # 如果没有使用目标级扰动，则计算特征级扰动的熵差
            original_entropy = calculate_entropy(features)
            perturbed_entropy = calculate_entropy(perturbed_features)
            entropy_diff = torch.abs(perturbed_entropy - original_entropy).mean().item()
            file_scores.append((img_name, entropy_diff))

    # 选择分数最高的前 n 张图片
    selected_files = [file for file, _ in sorted(file_scores, key=lambda x: -x[1])[:n]]

    # 复制选定的图像和标签到目标文件夹
    for file in tqdm(selected_files, desc="Copying selected files"):
        image_file = file
        label_file = file.replace('.jpg', '.txt')
        try:
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
            shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))
        except:
            print(f"Error copying {image_file} and {label_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply various perturbations to images, features, and model outputs.')
    parser.add_argument('--weights', type=str, default="weights/visdrone_best.pt", help='Path to the model weights file.')
    parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='Path to the dataset configuration file.')
    parser.add_argument('--source_images_folder', type=str, default="dataset/VisDrone/train/images", help='Path to the source images folder.')
    parser.add_argument('--source_labels_folder', type=str, default="dataset/VisDrone/train/labels", help='Path to the source labels folder.')
    parser.add_argument('--target_images_folder', type=str, default="dataset/VisDrone_part/init/images", help='Path to the target images folder.')
    parser.add_argument('--target_labels_folder', type=str, default="dataset/VisDrone_part/init/labels", help='Path to the target labels folder.')
    parser.add_argument('--copy_images_folder', type=str, default="dataset/VisDrone_part/image_perturbation_select/5/images", help='Path to the folder where selected images will be copied.')
    parser.add_argument('--copy_labels_folder', type=str, default="dataset/VisDrone_part/image_perturbation_select/5/labels", help='Path to the folder where selected labels will be copied.')
    parser.add_argument('--methods', nargs='+', default=[], help='Perturbation methods to apply. Options: flip, hsv, blur, noise, spatial_dropout, channel_wise_dropout, conf_noise')
    parser.add_argument('--num', type=int, default=500, help='Number of images to select after perturbations.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    model_backbone = DetectionModelBackboneFeature(cfg=args.data).to(device)
    model_combined = DetectionModelCombined(cfg=args.data).to(device)

    weights = args.weights
    ckpt = torch.load(attempt_download(weights), map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model_backbone.state_dict(), exclude=['anchor'])

    # Load weights into both models
    model_backbone.load_state_dict(csd, strict=False)
    model_combined.load_state_dict(csd, strict=False)

    model_backbone.eval()
    model_combined.eval()

    select_and_copy_files(args.source_images_folder, args.source_labels_folder, args.target_images_folder, args.target_labels_folder,
                          args.copy_images_folder, args.copy_labels_folder, model_backbone, model_combined, device, n=args.num,
                          perturbation_methods=args.methods)
