# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

class RGBTDataset(Dataset):
    """RGBT Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        thermal_path = img_item[3]

        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        thermal = read_image(thermal_path)
        if self.transform is not None: thermal = self.transform(thermal)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "thermals": thermal,
            "thermal_paths":thermal_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

class PredDataset(Dataset):

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform  = transform
        self.relabel = relabel

        sid_set = set()
        fid_set = set()
        pid_set = set()
        cam_set = set()
        for i in img_items:
            sid_set.add(i[1])
            fid_set.add(i[2])
            cam_set.add(i[3])
            pid_set.add(i[4])
            
        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        sid = img_item[1]
        fid = img_item[2]
        camid = img_item[3]
        pid = img_item[4]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "scenarios" : sid,
            "frames" : fid,
            "camids": camid,
            "targets": pid,
            "img_paths": img_path,
        }


