import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_model
from datagen import load_dataset, genxy
from utils import genanchors, loc2box2d
from datetime import datetime


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mode = 'train'
anno_file_path = '../datasets/widerface/'+mode+'.txt'
image_dir = '../datasets/widerface/'+mode
ishape = [512, 512, 3]
ssizes = [
	[128, 128],
	[64, 64], 
	[32, 32], 
	[16, 16],
]
asizes = [
	[[32, 32]],
	[[64, 64]],
	[[128, 128]],
	[[256, 256]],
]
total_classes = 1
iou_thresholds = [[0.5, 0.6], [0.45, 0.55], [0.4, 0.5], [0.35, 0.45]]
anchor_samplings = [256, 128, 64, 32]
total_examples = 100

a1box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0]), dtype='float32') # (h1*w1*k1, 4)
a2box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1]), dtype='float32') # (h2*w2*k2, 4)
a3box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2]), dtype='float32') # (h3*w3*k3, 4)
a4box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[3], asizes=asizes[3]), dtype='float32') # (h4*w4*k4, 4)
abox_2dtensors = [a1box_2dtensor, a2box_2dtensor, a3box_2dtensor, a4box_2dtensor]
abox_2dtensor = tf.concat(values=abox_2dtensors, axis=0)

dataset = load_dataset(anno_file_path=anno_file_path)
gen = genxy(
	dataset=dataset, 
	image_dir=image_dir, 
	ishape=ishape, 
	abox_2dtensors=abox_2dtensors, 
	iou_thresholds=iou_thresholds, 
	total_examples=total_examples,
	total_classes=total_classes, 
	anchor_samplings=anchor_samplings)

for _ in range(total_examples):
	print('{}: 1'.format(datetime.now().time()), end='\n')
	batchx_4dtensor, batchy_2dtensor, bboxes = next(gen)
	print('{}: 2'.format(datetime.now().time()), end='\n')
	pix = batchx_4dtensor[0]

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
	# print('{}: 3'.format(datetime.now().time()), end='\n')

	fg_bbox2d = fg_bbox_2dtensor.numpy()
	fg_clz1d = fg_clz_1dtensor.numpy()
	fg_abox2d = fg_abox_2dtensor.numpy()
	bg_abox2d = bg_abox_2dtensor.numpy()

	_, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(np.array(pix, dtype='uint8'))

	# print('{}: 4'.format(datetime.now().time()), end='\n')

	show_mode = [0, 2, 3]

	if 0 in show_mode:
		for i in range(len(bboxes)):
			box = bboxes[i][:4]
			clz = bboxes[i][4]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.8, 
				edgecolor='yellow',
				facecolor='none', 
				linestyle='-'))

	if 1 in show_mode:
		for i in range(fg_bbox2d.shape[0]):
			box = fg_bbox2d[i]
			clz = fg_clz1d[i]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			color = 'cyan' if clz == 0 else 'white'
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
			color = 'cyan' if clz == 0 else 'white'
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.6, 
				edgecolor=color,
				facecolor='none', 
				linestyle='-'))

	if 3 in show_mode:
		for i in range(bg_abox2d.shape[0]):
			box = bg_abox2d[i]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.2, 
				edgecolor='red',
				facecolor='none', 
				linestyle='-'))

	# print('{}: 5'.format(datetime.now().time()), end='\n')

	plt.show()














