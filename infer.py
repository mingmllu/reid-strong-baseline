# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from modeling import build_model
from torchvision import transforms
import cv2
import numpy as np




parser = argparse.ArgumentParser(description="ReID Baseline Inference")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("--paths_to_images", type=str, default="", help="CSV of paths to images")
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


if args.config_file != "":

    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()

if cfg.MODEL.DEVICE == "cuda":
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
cudnn.benchmark = False #True

use_gpu = torch.cuda.is_available()
# if use_gpu:
#     cudnn.benchmark = True

device = torch.device('cuda') if use_gpu else torch.device('cpu')

num_classes = 751
model = build_model(cfg, num_classes)
model.load_param(cfg.TEST.WEIGHT)
model.to(device)
model.eval()
for n, p in model.named_parameters():
    print(n, p.size())


list_paths = args.paths_to_images.split(',')

list_transforms = [
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
]

image_transform_pipeline = transforms.Compose(list_transforms)

def compute_features(img_roi):
    """
    Arguments:
    img_roi: 
        cropped image of region of interest in BGR
        type: Numpy ndarray of the shape HxWxC
    Return:
    Numpy array of size 1024
    """
    img_roi_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    img_roi_rgb = image_transform_pipeline(img_roi_rgb)
    img_roi_rgb = img_roi_rgb.unsqueeze(0)
    features = model(img_roi_rgb.to(device))
    features = features.cpu().data.numpy()
    features = features.squeeze()
    return features/np.linalg.norm(features)


for path in list_paths:
    if not os.path.isfile(path):
        break
    img = cv2.imread(path)
    embeddings = compute_features(img)
    title = path + '.baseline_ctr'
    np.save(title, embeddings)
    print("Saved", title + '.npy')
