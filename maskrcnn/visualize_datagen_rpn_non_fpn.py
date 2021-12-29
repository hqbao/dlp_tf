import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from datagen import genx, genx_selected, gety, genpy
from utils import genanchors, box2frame, bbe2box4d
from datetime import datetime


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

start_example_index = 0
num_of_examples 	= 100

asizes 				= [[91, 181], [128, 128], [181, 91]]
ishape 				= [1024, 1024, 3]
feature_map_size 	= [32, 32]
frame_mode 			= True
classes 			= ['face', 'none']
mapping 			= {0: 0}
iou_thresholds 		= [0.3, 0.5]
num_of_samples 		= 64
img_ids 			= None
ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

if img_ids is None:
	gen = genx(
		coco=coco, 
		img_dir=img_dir, 
		classes=classes, 
		limit=[start_example_index, start_example_index+num_of_examples],
		ishape=ishape)
else:
	gen = genx_selected(
		coco=coco, 
		img_dir=img_dir, 
		img_ids=img_ids,
		ishape=ishape)

for sample_order in range(num_of_examples):

	# generate x
	x, img_id = next(gen)

	# get labels
	bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

	_, ax = plt.subplots(figsize=(15, 7.35))

	# Generates anchors
	anchor4d = genanchors(
		isize=ishape[:2], 
		ssize=feature_map_size, # this is hardcode value, depends on pooling layers of base net
		asizes=asizes)
	anchor_4dtensor = tf.constant(value=anchor4d, dtype='float32')

	H = anchor4d.shape[0]
	W = anchor4d.shape[1]
	K = anchor4d.shape[2]

	# Generates y
	print('{}: START'.format(datetime.now().time()), end='\n')
	clzbbe_3dtensor = genpy(anchor_4dtensor=anchor_4dtensor, bbox2d=bbox2d[:, :4], iou_thresholds=iou_thresholds[:2], num_of_samples=num_of_samples)
	print('{}: END'.format(datetime.now().time()), end='\n')

	clz_3dtensor = clzbbe_3dtensor[:, :, :2*K]
	bbe_3dtensor = clzbbe_3dtensor[:, :, 2*K:]

	clz_4dtensor = tf.reshape(tensor=clz_3dtensor, shape=(H, W, K, 2))
	bbe_4dtensor = tf.reshape(tensor=bbe_3dtensor, shape=(H, W, K, 4))
	bbox_4dtensor = bbe2box4d(box_4dtensor=anchor_4dtensor, bbe_4dtensor=bbe_4dtensor)
	fg_3dtensor = clz_4dtensor[:, :, :, 0]
	bg_3dtensor = clz_4dtensor[:, :, :, 1]
	
	fg_indices = tf.where(condition=tf.math.equal(x=fg_3dtensor, y=1.0))
	fg_anchor_2dtensor = tf.gather_nd(params=anchor_4dtensor, indices=fg_indices)
	fg_2dtensor = tf.gather_nd(params=bbox_4dtensor, indices=fg_indices)
	
	bg_indices = tf.where(condition=tf.math.equal(x=bg_3dtensor, y=1.0))
	bg_anchor_2dtensor = tf.gather_nd(params=anchor_4dtensor, indices=bg_indices)

	ax.imshow(x/255)
	ax.set_xlabel('Order: {}, Image ID: {}'.format(sample_order, img_id))
	
	for i in range(fg_anchor_2dtensor.shape[0]):
		box = np.array(fg_anchor_2dtensor[i], dtype='float32')
		frame = box2frame(box=box, apoint=[0, 0])
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3],
			linewidth=0.8, 
			edgecolor='cyan',
			facecolor='none', 
			linestyle='-'))

		box = np.array(fg_2dtensor[i], dtype='float32')
		frame = box2frame(box=box, apoint=[0, 0])
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3],
			linewidth=1.0, 
			edgecolor='yellow',
			facecolor='none', 
			linestyle='-'))

	for i in range(bg_anchor_2dtensor.shape[0]):
		box = np.array(bg_anchor_2dtensor[i], dtype='float32')
		frame = box2frame(box=box, apoint=[0, 0])
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3],
			linewidth=0.3, 
			edgecolor='red',
			facecolor='none', 
			linestyle='-'))

	plt.show()









