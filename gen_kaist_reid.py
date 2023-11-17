from PIL import Image
import os
import os.path as osp
import json
import pandas as pd
import re
import numpy as np
import random
import shutil
from collections import defaultdict

def gen_reid_data(scenarios,
				  save_scenarios,
				  frames_root,
				  out_root,
				  sampling_rate = 23, 
				  img_width=1920, 
				  img_height=1080):
	pattern = re.compile(r'(s[\d]+)_(c[\d]+)')
	for s_id in scenarios: # [s01, s10, ..., s47]
		if s_id not in save_scenarios:
			continue
		out_dir = osp.join(out_root, s_id)
		if osp.exists(out_dir):
			shutil.rmtree(out_dir)
		os.makedirs(out_dir)

		ann_path = osp.join(ann_root, s_id)
		csv_names = os.listdir(ann_path)
		csv_names = [_ for _ in csv_names if 'csv' in _]
		for csv_name in csv_names: # ['s01_c01', s01_c02', ..., 's01_c16']
			_, c_id = pattern.search(csv_name).groups()
			# load anns
			anns = pd.read_csv(osp.join(ann_path, csv_name))
			f_ids = np.unique(anns['Frame#'])[::sampling_rate]
			for f_id in f_ids:
				f_anns = anns[anns['Frame#']==f_id]
				# load frame
				img_name = '%06d.jpg'%f_id
				img_path = osp.join(frames_root, s_id, c_id, 'rgb', img_name)
				img = Image.open(img_path)
				for idx, ann in f_anns.iterrows():
					# load ann
					p_id = int(ann['Person#'])
					x1 = max(0, int(ann['x']))
					y1 = max(0, int(ann['y']))
					x2 = min(img_width, x1 + int(ann['width']))
					y2 = min(img_height, y1 + int(ann['height']))
					# save crop farme
					width, height = (x2-x1), (y2-y1)
					if width <= 0 or height <= 0:
						continue
					if height / width >= 4.5:
						continue
					if width / height >= 2.5:
						continue
					crop_img = img.crop((x1, y1, x2, y2))
					out_img_name = f"{p_id:03}_{c_id}_{s_id}_f{f_id:04}.jpg"	
					out_path = osp.join(out_dir, out_img_name)
					crop_img.save(out_path)

def move_imgs(imgs, s_root, t_root):
	for img in imgs:
		s_path = osp.join(s_root, img)
		t_path = osp.join(t_root, img)
		shutil.copyfile(s_path, t_path)

def split_reid_data(scenarios, out_root):
	train_img_root = osp.join(out_root, 'train')
	query_img_root = osp.join(out_root, 'query')
	gallery_img_root = osp.join(out_root, 'gallery')

	if osp.exists(train_img_root):
		shutil.rmtree(train_img_root)
	if osp.exists(query_img_root):
		shutil.rmtree(query_img_root)
	if osp.exists(gallery_img_root):
		shutil.rmtree(gallery_img_root)

	os.makedirs(train_img_root)
	os.makedirs(query_img_root)
	os.makedirs(gallery_img_root)

	pattern = re.compile(r'([\d]+)_c([\d]+)')
	for s_id in scenarios:
		reid_img_root = osp.join(out_root, s_id)
		reid_imgs = os.listdir(reid_img_root)
	
		pid_to_imgs = defaultdict(list)
		pattern = re.compile(r'([\d]+)_c')
		for reid_img in reid_imgs:
			p_id = int(pattern.search(reid_img).groups()[0])
			pid_to_imgs[p_id].append(reid_img)
		
		pids = list(pid_to_imgs.keys())
		random.shuffle(pids)
		train_pids = pids[:len(pids)//2]
		test_pids = pids[len(pids)//2:]
		
		train_imgs = []
		for pid in train_pids:
			train_imgs.extend(pid_to_imgs[pid])

		test_query_imgs = []
		test_gallery_imgs = []
		pattern = re.compile(r'c([\d]+)')
		for pid in test_pids:
			test_imgs = pid_to_imgs[pid]
			cid_to_imgs = defaultdict(list)
			for test_img in test_imgs:
				cid = int(pattern.search(test_img).groups()[0])
				cid_to_imgs[cid].append(test_img)

			cids = list(cid_to_imgs.keys())
			if len(cids) == 1:
				for _ in cid_to_imgs.values():
					test_gallery_imgs.extend(_)
			else:
				assert len(cids) > 1
				for cid in cids:
					query_img = random.choice(cid_to_imgs[cid])
					query_idx = cid_to_imgs[cid].index(query_img)
					del cid_to_imgs[cid][query_idx]
					test_query_imgs.append(query_img)
				for _ in cid_to_imgs.values():
					test_gallery_imgs.extend(_)

		move_imgs(train_imgs, reid_img_root, train_img_root)
		move_imgs(test_query_imgs, reid_img_root, query_img_root)
		move_imgs(test_gallery_imgs, reid_img_root, gallery_img_root)

	rename_train_imgs(train_img_root)
	rename_test_imgs(query_img_root, gallery_img_root)

def rename_train_imgs(root):
	pattern = re.compile(r'([\d]+)_c[\d]+_s([\d]+)')
	imgs = os.listdir(root)
	id_info_map = defaultdict(int)
	count = 0
	for img in imgs:
		pid, sid = pattern.search(img).groups()
		info = (int(pid), int(sid))
		if  info not in id_info_map:
			id_info_map[info] = count
			count += 1
		new_pid = id_info_map[info]
		new_img = f'{new_pid:03d}_' + '_'.join(img.split('_')[1:])
		os.rename(osp.join(root, img), osp.join(root, new_img))

def rename_test_imgs(q_root, g_root):
	query_imgs = os.listdir(q_root)
	gallery_imgs = os.listdir(g_root)
	imgs = query_imgs + gallery_imgs
	pattern = re.compile(r'([\d]+)_c[\d]+_s([\d]+)')
	id_info_map = defaultdict(int)
	count = 0
	for img in imgs:
		pid, sid = pattern.search(img).groups()
		info = (int(pid), int(sid))
		if  info not in id_info_map:
			id_info_map[info] = count
			count += 1

	for q_img in query_imgs:
		pid, sid = pattern.search(q_img).groups()
		info = (int(pid), int(sid))
		new_pid = id_info_map[info]
		new_img = f'{new_pid:03d}_' + '_'.join(q_img.split('_')[1:])
		os.rename(osp.join(q_root, q_img), osp.join(q_root, new_img))

	for g_img in gallery_imgs:
		pid, sid = pattern.search(g_img).groups()
		info = (int(pid), int(sid))
		new_pid = id_info_map[info]
		new_img = f'{new_pid:03d}_' + '_'.join(g_img.split('_')[1:])
		os.rename(osp.join(g_root, g_img), osp.join(g_root, new_img))

if __name__ == '__main__':
	train_frames_root = '/home/miruware/shw/prj-mtmc/data/kaist_mtmdc/train'
	test_frames_root = '/home/miruware/shw/prj-mtmc/data/kaist_mtmdc/test'
	ann_root = '/home/miruware/shw/prj-mtmc/data/kaist_mtmdc/annotations'

	second = 3
	train_scenarios = os.listdir(train_frames_root); train_scenarios.sort();
	test_scenarios = os.listdir(test_frames_root); test_scenarios.sort();
	save_scenarios = ['s01', 's20', 's34', 's42', 's47'] 
	out_root = 'datasets/KaistMTMC-reID'
	# out_root = 'datasets/KaistMTMC-reID_RGBT'
	gen_reid_data(train_scenarios, save_scenarios, train_frames_root, out_root, sampling_rate=23*second)
	gen_reid_data(test_scenarios, save_scenarios, test_frames_root, out_root, sampling_rate=23*second)
	split_reid_data(save_scenarios, out_root)
	
