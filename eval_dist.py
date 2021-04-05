# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import numpy as np
import torchvision
from torch.backends import cudnn
from torchvision import datasets, transforms

import time
import os
import glob
import re
import sys
import os.path as osp

sys.path.append('.')
from config import cfg
from modeling import build_model
import cv2
import numpy as np

"""Image ReID"""

class Market1501(object):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

import os
from PIL import Image
import numpy as np
import os.path as osp
from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid




parser = argparse.ArgumentParser(description="ReID Baseline Inference")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("--paths_to_images", type=str, default="", help="CSV of paths to images")
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir',default='./data',type=str, help='dataset dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

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
print(device)

num_classes = 751
model = build_model(cfg, num_classes)
model.load_param(cfg.TEST.WEIGHT)
model.to(device)
model.eval()
# for n, p in model.named_parameters():
#     print(n, p.size())


######################################################################
# Load Data
# ---------

transform_val_list = [
    transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

dataset = Market1501(args.data_dir)

image_datasets = {}
image_datasets['train'] = ImageDataset(dataset.train, transform=transforms.Compose(transform_val_list))
image_datasets['val'] = ImageDataset(dataset.query, transform=transforms.Compose(transform_val_list))

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchsize,shuffle=True, num_workers=1) for x in ['train', 'val']}

num_classes = dataset.num_train_pids


which = "val"

dict_pid = {}
for data in dataloaders[which]:
    # get the inputs
    inputs, labels, _ = data
    inputs = inputs.to(device)
    features = model(inputs)
    features = features.cpu().data.numpy()
    features = np.transpose(np.divide(np.transpose(features), np.linalg.norm(features, axis=1)))
    for i, pid in enumerate(labels):
        pid = int(pid)
        if pid not in dict_pid:
            dict_pid[pid] = []
        dict_pid[pid].append(features[i])

# for pid in dict_pid:
#     u = np.vstack(dict_pid[pid])
#     print("Data array:", u.shape)
#     np.save(which+'_pid_%04d'%(pid), u)

list_vec_total = []
list_pid_total = []
for pid in dict_pid:
    list_vec_total.extend(dict_pid[pid])
    list_pid_total.extend([pid]*len(dict_pid[pid]))
v = np.vstack(list_vec_total)
i = np.array(list_pid_total)
np.save(which+'_market1501_data', v)
np.save(which+'_market1501_label', i)













