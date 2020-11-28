import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_model
from datagen import genxy
from utils import genanchors, loc2box2d
from datetime import datetime


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

anno_file_path = '../datasets/quizmarker/qm/train_anno.txt'
image_dir = '../datasets/quizmarker/qm/full_images'
ishape = [240, 200, 3]
ssize = [60, 50]
asizes = [[8, 8]]
total_classes = 2
iou_thresholds = [0.3, 0.35]
anchor_sampling = 1024
total_examples = 100

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
abox2d = abox4d.reshape((-1, 4))
abox_2dtensor = tf.constant(value=abox2d, dtype='float32')

gen = genxy(
	anno_file_path=anno_file_path, 
	image_dir=image_dir, 
	ishape=ishape, 
	abox_2dtensor=abox_2dtensor, 
	iou_thresholds=iou_thresholds, 
	total_classes=total_classes, 
	anchor_sampling=anchor_sampling,
	mode='train')

for _ in range(total_examples):
	print('{}: 1'.format(datetime.now().time()), end='\n')
	batchx_4dtensor, batchy_2dtensor, bboxes, _ = next(gen)
	print('{}: 2'.format(datetime.now().time()), end='\n')
	image = batchx_4dtensor[0]

	clz_2dtensor = batchy_2dtensor[:, :total_classes+1] # (h*w*k, total_classes+1)
	sum_clz_2dtensor = tf.math.reduce_sum(input_tensor=clz_2dtensor, axis=-1) # (h*w*k,)

	selected_indices = tf.where(condition=tf.math.equal(x=sum_clz_2dtensor, y=1))
	foreground_indices = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.equal(x=sum_clz_2dtensor, y=1),
			y=tf.math.not_equal(x=clz_2dtensor[:, -1], y=1)))
	background_indices = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.equal(x=sum_clz_2dtensor, y=1),
			y=tf.math.equal(x=clz_2dtensor[:, -1], y=1)))
	
	fg_abox_2dtensor = tf.gather_nd(params=abox_2dtensor, indices=foreground_indices) # (unknow, 4)
	fg_clzloc_2dtensor = tf.gather_nd(params=batchy_2dtensor, indices=foreground_indices) # (unknow, total_classes+1+4)
	fg_clz_2dtensor = fg_clzloc_2dtensor[:, :total_classes+1] # (unknow, total_classes+1)
	fg_clz_1dtensor = tf.math.argmax(input=fg_clz_2dtensor, axis=-1) # (unknow,)
	fg_loc_2dtensor = fg_clzloc_2dtensor[:, total_classes+1:] # (unknow, 4)
	fg_bbox_2dtensor = loc2box2d(box_2dtensor=fg_abox_2dtensor, bbe_2dtensor=fg_loc_2dtensor)
	
	bg_abox_2dtensor = tf.gather_nd(params=abox_2dtensor, indices=background_indices) # (unknow, 4)
	print('{}: 3'.format(datetime.now().time()), end='\n')

	fg_bbox2d = fg_bbox_2dtensor.numpy()
	fg_clz1d = fg_clz_1dtensor.numpy()
	fg_abox2d = fg_abox_2dtensor.numpy()
	bg_abox2d = bg_abox_2dtensor.numpy()

	_, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(image)

	print('{}: 4'.format(datetime.now().time()), end='\n')

	show_mode = []

	if 0 in show_mode:
		for i in range(len(bboxes)):
			box = bboxes[i][:4]
			clz = bboxes[i][4]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.8, 
				edgecolor='black',
				facecolor='none', 
				linestyle='-'))

	if 1 in show_mode:
		for i in range(fg_bbox2d.shape[0]):
			box = fg_bbox2d[i]
			clz = fg_clz1d[i]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			color = 'cyan' if clz == 0 else 'yellow'
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.8, 
				edgecolor=color,
				facecolor='none', 
				linestyle='-'))
	
	if 2 in show_mode:
		for i in range(fg_abox2d.shape[0]):
			box = fg_abox2d[i]
			clz = fg_clz1d[i]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			color = 'cyan' if clz == 0 else 'yellow'
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.8, 
				edgecolor=color,
				facecolor='none', 
				linestyle='-'))

	if 3 in show_mode:
		for i in range(bg_abox2d.shape[0]):
			box = bg_abox2d[i]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.1, 
				edgecolor='red',
				facecolor='none', 
				linestyle='-'))

	print('{}: 5'.format(datetime.now().time()), end='\n')

	plt.show()














