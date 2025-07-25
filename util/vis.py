import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
from matplotlib import pyplot as plt

#from ..model.mambaad import MambaUPNet

# b.py
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path("/home/gpu/ADer-main")
sys.path.append(str(project_root))

from model.mambaad import MambaUPNet

import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color
import torch.nn.functional as F

def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps, prototype_distance256, original_anomaly_map, fs_list, method, root_out, dataset_name,
                   self=None):


    fs = fs_list[0]

    # 上采样到[16, 64, 256, 256]
    #upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    #fs = upsample(fs)

    # 通道压缩到1
    conv = nn.Conv2d(64, 1, kernel_size=1)  # 1x1卷积，用于通道转换
    #conv = conv.cuda()
    #fs = conv(fs)

    mamba = MambaUPNet()
    mamba.vis_conv = mamba.vis_conv.cuda()
    fs = mamba.vis_conv(fs)

    # 逆转归一化
    #imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand_as(fs)
    #imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand_as(fs)
    #imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=fs.device)
    #imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=fs.device)
    #fs = fs.cpu()
    #fs = (fs * imagenet_std + imagenet_mean)
    #print(fs.size())


    # 确保值的范围在0到255之间
    #fs = fs.clamp(0, 255).type(torch.uint8)
    #fs = fs.cuda()



    #print(fs.size())
    #print(1111111111111)

    #for element in fs_list:
    #    print(len(element))
    #print(len(fs_list))
    #fs = fs.cuda()
    #fs_list = fs_list.cuda()
    #fs_np = np.array(fs_list)
    #fs_np = fs_np.cuda()
    #print(fs_np.shape)
    #print(imgs.size())
    #fs_list = torch.tensor(fs_list)
    #print(fs_list.size())



    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)
    if fs.shape[-1] != img_masks.shape[-1]:
        fs = F.interpolate(fs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)
    for idx, (img_path, img, img_mask, anomaly_map, prototype_distance256, original_anomaly_map, fs) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps, prototype_distance256, original_anomaly_map, fs)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        img_path = f'{out_dir}/{img_num}_img.png'
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)


        img_rec = img * std[:, None, None] + mean[:, None, None]
        # RGB image
        img_rec = Image.fromarray((img_rec * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        img_rec.save(img_path)

        # RGB image with anomaly map
        anomaly_map = anomaly_map / anomaly_map.max()
        anomaly_map = cm.jet(anomaly_map)
        # anomaly_map = cm.rainbow(anomaly_map)
        anomaly_map = (anomaly_map[:, :, :3] * 255).astype('uint8')
        anomaly_map = Image.fromarray(anomaly_map)
        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)


        # mask
        img_mask = Image.fromarray((img_mask * 255).astype(np.uint8).transpose(1, 2, 0).repeat(3, axis=2))
        img_mask.save(mask_path)