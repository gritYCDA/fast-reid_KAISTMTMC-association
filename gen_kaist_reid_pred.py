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


if __name__ == '__main__':
	test_frames_root = '/home/miruware/shw/prj-mtmc/data/kaist_mtmdc/test'
	pred_root = '/home/miruware/shw/prj-mtmc/work_dirs/KAIST_MTMDC/qdtrack-frcnn_r50_kaist_mtmdc_test/track'
	out_root = 'datasets/KaistMTMC-reID/pred'
	test_scenarios = os.listdir(test_frames_root); test_scenarios.sort()
	sampling_rate = 23

	if osp.exists(out_root):
		shutil.rmtree(out_root)
	os.makedirs(out_root)
	for s_id in test_scenarios:
		cameras = os.listdir(osp.join(pred_root, s_id)); cameras.sort()
		cameras = [_[:3] for _ in cameras]
		for c_id in cameras:
			pred_path = osp.join(pred_root, s_id, f'{c_id}.txt')
			imgs_path = osp.join(test_frames_root, s_id, c_id, 'rgb')
			with open(pred_path) as f:
				preds = f.readlines()

			preds_by_frame = defaultdict(list)
			for pred in preds:
				pred = pred.split(',')[:-3]
				pred = [float(_) for _ in pred]
				f_id, p_id, x1, y1, w, h, conf = pred
				#
				f_id = int(f_id-1) * sampling_rate
				p_id = int(p_id)
				x1 = max(0, int(x1))
				y1 = max(0, int(y1))
				x2 = min(1920, x1 + int(w))
				y2 = min(1080, y1 + int(h))
				preds_by_frame[f_id].append([p_id, x1, y1, x2, y2, conf])
				
			for f_id, preds in preds_by_frame.items():
				img_path = osp.join(imgs_path, f'{f_id:06}.jpg')
				img = Image.open(img_path)
				for pred in preds:
					p_id, x1, y1, x2, y2, conf = pred
					crop_img = img.crop((x1, y1, x2, y2))
					out_img = f"{s_id}_{f_id}_{c_id}_{p_id}.jpg"
					out_path = osp.join(out_root, out_img)
					crop_img.save(out_path)