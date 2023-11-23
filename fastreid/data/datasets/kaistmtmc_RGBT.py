import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class KaistMTMCRGBT(ImageDataset):

    dataset_dir = 'KaistMTMC-reID_RGBT'
    dataset_name = 'KaistMTMCRGBT'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train') 
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(KaistMTMCRGBT, self).__init__(train, query, gallery, **kwargs)

    ## RGBT version
    def process_dir(self, img_dir_path, is_train=True):
        img_paths = glob.glob(osp.join(img_dir_path, 'rgb', '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([\d]+)')

        data = []
        for img_path in img_paths:
            thermal_path = osp.join(img_dir_path, 'thermal', img_path.split('/')[-1])
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 16
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, thermal_path))
        return data