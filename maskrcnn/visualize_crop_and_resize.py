import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from datagen import genx, gety, gendy
from utils import normbox2d, pad_roi2d

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

start_example_index = 0
num_of_examples = 100

asizes = [	
	[[32, 32]],
	[[64, 64]],
	[[128, 128]],
	[[256, 256]],
]
ishape = [1024, 1024, 3]
feature_map_sizes = [[128, 128], [64, 64], [32, 32], [16, 16]]
frame_mode = True
classes = ['face', 'none']
mapping = {0: 0}
iou_thresholds = [0.3, 0.5]
max_num_of_rois = 7
unified_roi_size = [7, 7]
k0 = 5
ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=[start_example_index, start_example_index+num_of_examples],
	ishape=ishape)

for batch_idx in range(num_of_examples):

	# generate x
	x, img_id = next(gen)

	fig, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(x/255)
		plt.show()

	# get labels
	bbox2d, tmasks = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

	# predict rois
	roi2d = pad_roi2d(roi2d=bbox2d[:, :4], max_num_of_rois=max_num_of_rois, pad_roi=[0, 0, ishape[0], ishape[1]])
	roi_2dtensor = tf.constant(value=roi2d, dtype='float32')

	# feature map
	image = tf.constant(value=x, dtype='float32') # (128, 256, 3)
	images = tf.expand_dims(input=image, axis=0) # (batch_size, 128, 256, 3), batch_size = 1

	# feature_map1 = tf.image.resize(images=images, size=tf.constant(value=feature_map_sizes[0], dtype='int32'))
	# feature_map2 = tf.image.resize(images=images, size=tf.constant(value=feature_map_sizes[1], dtype='int32'))
	# feature_map3 = tf.image.resize(images=images, size=tf.constant(value=feature_map_sizes[2], dtype='int32'))
	# feature_map4 = tf.image.resize(images=images, size=tf.constant(value=feature_map_sizes[3], dtype='int32'))

	# feature_map1 = tf.nn.max_pool2d(input=images, ksize=1, strides=2, padding='VALID')
	# feature_map2 = tf.nn.max_pool2d(input=feature_map1, ksize=1, strides=2, padding='VALID')
	# feature_map3 = tf.nn.max_pool2d(input=feature_map2, ksize=1, strides=2, padding='VALID')
	# feature_map4 = tf.nn.max_pool2d(input=feature_map3, ksize=1, strides=2, padding='VALID')

	# feature_map1 = tf.nn.avg_pool2d(input=images, ksize=2, strides=1, padding='VALID')
	# feature_map2 = tf.nn.avg_pool2d(input=feature_map1, ksize=2, strides=1, padding='VALID')
	# feature_map3 = tf.nn.avg_pool2d(input=feature_map2, ksize=2, strides=1, padding='VALID')
	# feature_map4 = tf.nn.avg_pool2d(input=feature_map3, ksize=2, strides=1, padding='VALID')

	feature_map1 = tf.nn.avg_pool2d(input=images,       ksize=2, strides=1, padding='VALID')
	feature_map2 = tf.nn.avg_pool2d(input=feature_map1, ksize=2, strides=1, padding='VALID')
	feature_map3 = tf.nn.avg_pool2d(input=feature_map2, ksize=2, strides=1, padding='VALID')
	feature_map4 = tf.nn.avg_pool2d(input=feature_map3, ksize=2, strides=1, padding='VALID')

	feature_map_4dtensors = [feature_map1, feature_map2, feature_map3, feature_map4]
	crop_size = unified_roi_size

	y1, x1, y2, x2 = tf.split(value=roi_2dtensor, num_or_size_splits=4, axis=1) # (num_of_rois, 1)
	h = y2 - y1
	w = x2 - x1
	roi_level_2dtensor = tf.math.log(tf.sqrt(h*w))/tf.math.log(2.0)
	roi_level_2dtensor = tf.math.minimum(3, tf.math.maximum(0, tf.cast(tf.math.round(roi_level_2dtensor - k0), dtype='int32'))) # (num_of_rois, 1)
	roi_level_2dtensor = tf.squeeze(input=roi_level_2dtensor, axis=1) # (num_of_rois,)
	norm_roi_2dtensor = normbox2d(box_2dtensor=roi_2dtensor, isize=ishape[:2])

	resized_roi_4dtensors = []

	for lvl in range(4):
		feature_map_4dtensor = feature_map_4dtensors[lvl]
		roi_indices = tf.where(condition=tf.math.equal(x=roi_level_2dtensor, y=lvl))
		lvl_roi_2dtensor = tf.gather_nd(params=norm_roi_2dtensor, indices=roi_indices)
		print(lvl_roi_2dtensor)
		roi_indices *= 0 # batch_size = 1, indices should be all zeros
		roi_indices = tf.cast(x=roi_indices[:, 0], dtype='int32')

		# Stop gradient propogation to ROIs
		lvl_roi_2dtensor = tf.stop_gradient(lvl_roi_2dtensor)
		roi_indices = tf.stop_gradient(roi_indices)

		resized_roi_4dtensor = tf.image.crop_and_resize(
			image=feature_map_4dtensor,
			boxes=lvl_roi_2dtensor,
			box_indices=roi_indices,
			crop_size=tf.constant(value=crop_size),
			method="bilinear")

		resized_roi_4dtensors.append(resized_roi_4dtensor)

	resized_roi_4dtensor = tf.concat(values=resized_roi_4dtensors, axis=0) # (num_of_rois, crop_size_height, crop_size_width, channels)

	for roi_idx in range(resized_roi_4dtensor.shape[0]):

		roi_3dtensor = resized_roi_4dtensor[roi_idx] # (crop_size_height, crop_size_width, channels)
		
		fig, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(roi_3dtensor/255)
		ax.set_xlim([0, roi_3dtensor.shape[1]])
		ax.set_ylim([roi_3dtensor.shape[0], 0])
		plt.show()








