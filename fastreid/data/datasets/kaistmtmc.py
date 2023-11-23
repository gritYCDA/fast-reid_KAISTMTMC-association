import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
# class KaistMTMC(ImageDataset):

# 	dataset_dir = 'KaistMTMC-reID'
# 	dataset_name = 'kaistmtmc'

# 	dataset_dir = 'KaistMTMC-reID-6s'

# 	def __init__(self, root='datasets', **kwargs):
# 		self.root = root
# 		self.dataset_dir = osp.join(self.root, self.dataset_dir)
# 		# self.train_dir = osp.join(self.dataset_dir, 'train') 
# 		self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
# 		self.query_dir = osp.join(self.dataset_dir, 'query')
# 		# self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
# 		self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

# 		required_files = [
# 			self.dataset_dir,
# 			self.train_dir,
# 			self.query_dir,
# 			self.gallery_dir
# 		]

# 		train = self.process_dir(self.train_dir)
# 		query = self.process_dir(self.query_dir, is_train=False)
# 		gallery = self.process_dir(self.gallery_dir, is_train=False)

# 		super(KaistMTMC, self).__init__(train, query, gallery, **kwargs)

# 	def process_dir(self, dir_path, is_train=True):
# 		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
# 		pattern = re.compile(r'([-\d]+)_c([\d]+)')

# 		data = []
# 		for img_path in img_paths:
# 			pid, camid = map(int, pattern.search(img_path).groups())
# 			assert 1 <= camid <= 16
# 			camid -= 1  # index starts from 0
# 			if is_train:
# 				pid = self.dataset_name + "_" + str(pid)
# 				camid = self.dataset_name + "_" + str(camid)
# 			data.append((img_path, pid, camid))

# 		return data

# @DATASET_REGISTRY.register()
# class KaistMTMC(ImageDataset):

# 	dataset_dir = 'KaistMTMC-reID'
# 	dataset_name = 'kaistmtmc'

# 	dataset_dir = 'KaistMTMC-reID-6s'

# 	def __init__(self, root='datasets', **kwargs):
# 		self.root = root
# 		self.dataset_dir = osp.join(self.root, self.dataset_dir)
# 		self.train_dir = osp.join(self.dataset_dir, 'train') 
# 		self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
# 		self.query_dir = osp.join(self.dataset_dir, 'query')
# 		# self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
# 		self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

# 		required_files = [
# 			self.dataset_dir,
# 			self.train_dir,
# 			self.query_dir,
# 			self.gallery_dir
# 		]

# 		train = self.process_dir(self.train_dir)
# 		query = self.process_dir(self.query_dir, is_train=False)
# 		gallery = self.process_dir(self.gallery_dir, is_train=False)

# 		super(KaistMTMC, self).__init__(train, query, gallery, **kwargs)

# 	def process_dir(self, dir_path, is_train=True):
# 		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
# 		pattern = re.compile(r'([-\d]+)_c([\d]+)')

# 		data = []
# 		for img_path in img_paths:
# 			pid, camid = map(int, pattern.search(img_path).groups())
# 			assert 1 <= camid <= 16
# 			camid -= 1  # index starts from 0
# 			if is_train:
# 				pid = self.dataset_name + "_" + str(pid)
# 				camid = self.dataset_name + "_" + str(camid)
# 			data.append((img_path, pid, camid))
# 		return data
	

@DATASET_REGISTRY.register()
class KaistMTMC(ImageDataset):

	dataset_dir = 'KaistMTMC-reID_RGBT'
	dataset_name = 'kaistmtmc'

	def __init__(self, root='datasets', **kwargs):
		self.root = root
		self.dataset_dir = osp.join(self.root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train/rgb') 
		self.query_dir = osp.join(self.dataset_dir, 'query/rgb')
		self.gallery_dir = osp.join(self.dataset_dir, 'gallery/rgb')

		required_files = [
			self.dataset_dir,
			self.train_dir,
			self.query_dir,
			self.gallery_dir
		]

		train = self.process_dir(self.train_dir)
		query = self.process_dir(self.query_dir, is_train=False)
		gallery = self.process_dir(self.gallery_dir, is_train=False)

		super(KaistMTMC, self).__init__(train, query, gallery, **kwargs)

	def process_dir(self, dir_path, is_train=True):
		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
		pattern = re.compile(r'([-\d]+)_c([\d]+)')

		data = []
		for img_path in img_paths:
			pid, camid = map(int, pattern.search(img_path).groups())
			assert 1 <= camid <= 16
			camid -= 1  # index starts from 0
			if is_train:
				pid = self.dataset_name + "_" + str(pid)
				camid = self.dataset_name + "_" + str(camid)
			data.append((img_path, pid, camid))
		return data