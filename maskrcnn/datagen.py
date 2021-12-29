import numpy as np
import skimage.io as io
import cv2
import tensorflow as tf

from pycocotools.coco import COCO
from random import shuffle
from utils import combbe4d, combbe2d, comiou4d, comiou2d


def genx(coco, img_dir, classes, limit, ishape):
	'''
	Arguments
		coco:
		img_dir:
		classes:
		limit:
		ishape:

	Return 
		x:
		img_id:
	'''

	cat_ids = coco.getCatIds(catNms=classes)
	img_ids = coco.getImgIds(catIds=cat_ids)
	# img_ids = coco.getImgIds();
	imgs = coco.loadImgs(img_ids)
	shuffle(imgs)

	print(cat_ids)
	print(len(imgs))

	for img in imgs[limit[0]:limit[1]]:

		# image data (h, w, channels)
		pix = io.imread('{}/{}'.format(img_dir, img['file_name']))

		# padding input img 
		x = np.zeros(ishape, dtype='float32')
		if len(pix.shape) == 2:
			x[:pix.shape[0], :pix.shape[1], 0] = pix[:ishape[0], :ishape[1]]
			x[:pix.shape[0], :pix.shape[1], 1] = pix[:ishape[0], :ishape[1]]
			x[:pix.shape[0], :pix.shape[1], 2] = pix[:ishape[0], :ishape[1]]
		else:
			x[:pix.shape[0], :pix.shape[1], :] = pix[:ishape[0], :ishape[1], :]

		yield x, img['id']

def genx_selected(coco, img_dir, img_ids, ishape):
	'''
	Arguments
		coco:
		img_dir:
		selection:
		ishape:

	Return 
		x:
		img_id:
	'''

	imgs = coco.loadImgs(img_ids)

	for img in imgs:

		# image data (h, w, channels)
		pix = io.imread('{}/{}'.format(img_dir, img['file_name']))

		# padding input img 
		x = np.zeros(ishape, dtype='float32')
		if len(pix.shape) == 2:
			x[:pix.shape[0], :pix.shape[1], 0] = pix[:ishape[0], :ishape[1]]
			x[:pix.shape[0], :pix.shape[1], 1] = pix[:ishape[0], :ishape[1]]
			x[:pix.shape[0], :pix.shape[1], 2] = pix[:ishape[0], :ishape[1]]
		else:
			x[:pix.shape[0], :pix.shape[1], :] = pix[:ishape[0], :ishape[1], :]

		yield x, img['id']

def gety(coco, img_id, classes, frame_mode=False, mapping=None):
	'''
	Arguments
		coco:
		img_id:
		classes:
		frame_mode: True is frame mode, False is box mode
		mapping:
	Return 
		bbox2d:
		masks:
	'''

	cat_ids = coco.getCatIds(catNms=classes)
	ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=0)
	anns = coco.loadAnns(ids=ann_ids)
	masks = None
	# masks = np.array([coco.annToMask(ann) for ann in anns])

	if mapping is None:
		if frame_mode:
			return np.array([[ann['bbox'][1], ann['bbox'][0], 
				ann['bbox'][1]+ann['bbox'][3], ann['bbox'][0]+ann['bbox'][2], 
				ann['category_id']] for ann in anns]), masks

		return np.array([[ann['bbox'][0], ann['bbox'][1], 
			ann['bbox'][2], ann['bbox'][3], 
			ann['category_id']] for ann in anns]), masks

	else:
		if frame_mode:
			return np.array([[ann['bbox'][1], ann['bbox'][0], 
				ann['bbox'][1]+ann['bbox'][3], ann['bbox'][0]+ann['bbox'][2], 
				mapping[ann['category_id']]] for ann in anns]), masks

		return np.array([[ann['bbox'][0], ann['bbox'][1], 
			ann['bbox'][2], ann['bbox'][3], 
			mapping[ann['category_id']]] for ann in anns]), masks

def genpy(anchor_4dtensor, bbox2d, iou_thresholds, num_of_samples):
	'''
	Arguments
		anchor_4dtensor: shape (h, w, k, [y1, x1, y2, x2])
		bbox2d: shape (num_of_bboxes, [y1, x1, y2, x2])
		iou_thresholds: [background_max_iou, foreground_min_iou]
		num_of_samples:
	Return 
		clzbbe_3dtensor
			clz_3dtensor: shape (h, w, 2k), 2k class {background, foreground}
			bbe_3dtensor: shape (h, w, 4k), 4k regession {ty, tx, th, tw}
	'''

	H = anchor_4dtensor.shape[0]
	W = anchor_4dtensor.shape[1]
	K = anchor_4dtensor.shape[2]
	num_of_bboxes = bbox2d.shape[0]

	# Creates bbox_4dtensors
	bbox5d = np.zeros((num_of_bboxes, H, W, K, 4), dtype='float32')
	for i in range(num_of_bboxes):
		bbox5d[i, :, :, :, 0] = bbox2d[i, 0]
		bbox5d[i, :, :, :, 1] = bbox2d[i, 1]
		bbox5d[i, :, :, :, 2] = bbox2d[i, 2]
		bbox5d[i, :, :, :, 3] = bbox2d[i, 3]

	bbox_4dtensors = []
	for i in range(num_of_bboxes):
		bbox_4dtensors.append(tf.constant(value=bbox5d[i], dtype='float32'))

	# Computes iou_tensors, bbe_4dtensors
	iou_4dtensors = []
	bbe_4dtensors = []
	for i in range(num_of_bboxes):
		iou_4dtensor = comiou4d(bbox_4dtensor=bbox_4dtensors[i], anchor_4dtensor=anchor_4dtensor) # (h, w, k, 1)
		bbe_4dtensor = combbe4d(bbox_4dtensor=bbox_4dtensors[i], anchor_4dtensor=anchor_4dtensor) # (h, w, k, 4)
		iou_4dtensors.append(iou_4dtensor)
		bbe_4dtensors.append(bbe_4dtensor)

	# Assigns positive, neutral, negative labels to anchors
	zero_4dtensor = tf.zeros(shape=(H, W, K, 1), dtype='float32') # (h, w, k, 1)
	one_4dtensor = tf.ones(shape=(H, W, K, 1), dtype='float32') # (h, w, k, 1)
	zero_zero_4dtensor = tf.concat(values=[zero_4dtensor, zero_4dtensor], axis=3) # (h, w, k, 2)
	zero_one_4dtensor = tf.concat(values=[zero_4dtensor, one_4dtensor], axis=3) # (h, w, k, 2)
	one_zero_4dtensor = tf.concat(values=[one_4dtensor, zero_4dtensor], axis=3) # (h, w, k, 2)
	iou_threshold_tensor = tf.constant(value=iou_thresholds, dtype='float32') # (2,)

	pos_5dtensors = []
	neg_5dtensors = []
	for i in range(num_of_bboxes):
		# Disable neutral & negative anchors
		pos_4dtensor = tf.where(
			condition=tf.math.greater_equal(x=iou_4dtensors[i], y=iou_threshold_tensor[1]),
			x=iou_4dtensors[i],
			y=0) # (h, w, k, 1)
		pos_5dtensor = tf.expand_dims(input=pos_4dtensor, axis=0) # (1, h, w, k, 1)
		pos_5dtensors.append(pos_5dtensor)

		# Disable neutral and positive anchors
		neg_4dtensor = tf.where(
			condition=tf.math.less_equal(x=iou_4dtensors[i], y=iou_threshold_tensor[0]),
			x=iou_4dtensors[i],
			y=num_of_bboxes) # (h, w, k, 1)
		neg_5dtensor = tf.expand_dims(input=neg_4dtensor, axis=0) # (1, h, w, k, 1)
		neg_5dtensors.append(neg_5dtensor) 

	pos_5dtensor = tf.concat(values=pos_5dtensors, axis=0) # (num_of_bboxes, h, w, k, 1)
	neg_5dtensor = tf.concat(values=neg_5dtensors, axis=0) # (num_of_bboxes, h, w, k, 1)

	max_pos_4dtensor = tf.reduce_max(input_tensor=pos_5dtensor, axis=0) # (h, w, k, 1)
	avg_neg_4dtensor = tf.reduce_mean(input_tensor=neg_5dtensor, axis=0) # (h, w, k, 1)

	# Make positive & negative balanced
	max_pos4d = np.array(max_pos_4dtensor)
	avg_neg4d = np.array(avg_neg_4dtensor)
	max_pos1d = max_pos4d.reshape(-1)
	avg_neg1d = avg_neg4d.reshape(-1)
	pos_indices, = np.where(max_pos1d >= iou_threshold_tensor[1])
	neg_indices, = np.where(np.logical_and(
		avg_neg1d >= 0,
		avg_neg1d <= iou_threshold_tensor[0]))
	num_of_pos = len(pos_indices)
	num_of_neg = len(neg_indices)
	keep = min(min(num_of_pos, num_of_neg), num_of_samples)

	## Select positives randomly
	np.random.shuffle(pos_indices)
	pos_indices = pos_indices[keep:]
	for pos_idx in pos_indices:
		max_pos1d[pos_idx] = 0

	## Select nagatives (more hard negatives)
	filter_avg_neg1d = avg_neg1d[neg_indices]
	neg_indices = [x for _,x in sorted(zip(filter_avg_neg1d, neg_indices), reverse=True)]
	hard_neg1d, = np.where(np.logical_and(
		avg_neg1d > 0,
		avg_neg1d <= iou_threshold_tensor[0]))
	num_of_hard_neg = len(hard_neg1d)
	### Shuffle easy negatives
	easy_neg_indices = neg_indices[num_of_hard_neg:]
	np.random.shuffle(easy_neg_indices)
	neg_indices[num_of_hard_neg:] = easy_neg_indices
	### Shuffle hard + part of easy negatives
	hard_and_easy_neg_indices = neg_indices[:2*num_of_hard_neg]
	np.random.shuffle(hard_and_easy_neg_indices)
	neg_indices[:2*num_of_hard_neg] = hard_and_easy_neg_indices
	### Shuffle part of easy negatives
	part_of_easy_neg_indices = neg_indices[2*num_of_hard_neg:]
	np.random.shuffle(part_of_easy_neg_indices)
	neg_indices[2*num_of_hard_neg:] = part_of_easy_neg_indices

	## Select negatives
	if keep < num_of_samples:
		keep += num_of_samples - keep # pad negatives if num of positives < num_of_samples
	neg_indices = neg_indices[keep:]
	for neg_idx in neg_indices:
		avg_neg1d[neg_idx] = num_of_bboxes

	max_pos4d = max_pos1d.reshape(max_pos4d.shape)
	avg_neg4d = avg_neg1d.reshape(avg_neg4d.shape)

	max_pos_4dtensor = tf.constant(value=max_pos4d, dtype='float32')
	avg_neg_4dtensor = tf.constant(value=avg_neg4d, dtype='float32')

	# Assign clz to foreground anchors
	foreground_4dtensor = tf.where(
		condition=tf.math.greater_equal(x=max_pos_4dtensor, y=iou_threshold_tensor[1]),
		x=one_zero_4dtensor,
		y=zero_zero_4dtensor) # (h, w, k, 2)

	# Assign clz to foreground and background anchors
	clz_4dtensor = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.greater_equal(x=avg_neg_4dtensor, y=0),
			y=tf.math.less_equal(x=avg_neg_4dtensor, y=iou_threshold_tensor[0])),
		x=zero_one_4dtensor,
		y=foreground_4dtensor) # (h, w, k, 2)

	# Reduce with max iou condition
	bbe_4dtensor = bbe_4dtensors[0]
	iou_4dtensor = iou_4dtensors[0]
	for i in range(1, num_of_bboxes):
		bbe_4dtensor = tf.where(
			condition=tf.math.greater(x=iou_4dtensors[i], y=iou_4dtensor),
			x=bbe_4dtensors[i],
			y=bbe_4dtensor) # (h, w, k, 4)
		iou_4dtensor = tf.where(
			condition=tf.math.greater(x=iou_4dtensors[i], y=iou_4dtensor),
			x=iou_4dtensors[i],
			y=iou_4dtensor) # (h, w, k, 1)

	clz_3dtensor = tf.reshape(tensor=clz_4dtensor, shape=[H, W, 2*K]) # (h, w, 2k)
	bbe_3dtensor = tf.reshape(tensor=bbe_4dtensor, shape=[H, W, 4*K]) # (h, w, 4k)
	clzbbe_3dtensor = tf.concat(values=[clz_3dtensor, bbe_3dtensor], axis=2)

	return clzbbe_3dtensor

def gendy(num_of_classes, roi_2dtensor, bbox2d, unified_roi_size, isize, iou_thresholds):
	'''
	Arguments
		num_of_classes:
		roi_2dtensor:
		bbox2d: shape (num_of_rois, [y1, x1, y2, x2, class])
		unified_roi_size:
		isize: image size:
		iou_thresholds:
	Return 
		clzbbe_2dtensor: (num_of_rois, num_of_class+4)
	'''

	num_of_rois = roi_2dtensor.shape[0]
	num_of_bboxes = bbox2d.shape[0]
	iou_threshold_tensor = tf.constant(value=iou_thresholds, dtype='float32') # (2,)

	# Create bbox_2dtensors
	bbox3d = np.zeros((num_of_bboxes, num_of_rois, 5), dtype='float32')
	for i in range(num_of_bboxes):
		bbox3d[i, :, 0] = bbox2d[i, 0]
		bbox3d[i, :, 1] = bbox2d[i, 1]
		bbox3d[i, :, 2] = bbox2d[i, 2]
		bbox3d[i, :, 3] = bbox2d[i, 3]
		bbox3d[i, :, 4] = bbox2d[i, 4]

	bbox_2dtensors = []
	for i in range(num_of_bboxes):
		bbox_2dtensors.append(tf.constant(value=bbox3d[i], dtype='float32'))

	# Reduce catagory, bbe by highest iou
	best_cat_2dtensor = bbox_2dtensors[0][:, 4:5] # (num_of_rois, 1)
	best_bbe_2dtensor = combbe2d(bbox_2dtensor=bbox_2dtensors[0][:, :4], roi_2dtensor=roi_2dtensor) # (num_of_rois, 4)
	best_iou_2dtensor = comiou2d(bbox_2dtensor=bbox_2dtensors[0][:, :4], roi_2dtensor=roi_2dtensor) # (num_of_rois, 1)
	for i in range(1, num_of_bboxes):
		cat_2dtensor = bbox_2dtensors[i][:, 4:5] # (num_of_rois, 1)
		bbe_2dtensor = combbe2d(bbox_2dtensor=bbox_2dtensors[i][:, :4], roi_2dtensor=roi_2dtensor) # (num_of_rois, 4)
		iou_2dtensor = comiou2d(bbox_2dtensor=bbox_2dtensors[i][:, :4], roi_2dtensor=roi_2dtensor) # (num_of_rois, 1)
		best_cat_2dtensor = tf.where(
			condition=tf.math.greater(x=iou_2dtensor, y=best_iou_2dtensor),
			x=cat_2dtensor,
			y=best_cat_2dtensor) # (num_of_rois, 1)
		best_bbe_2dtensor = tf.where(
			condition=tf.math.greater(x=iou_2dtensor, y=best_iou_2dtensor),
			x=bbe_2dtensor,
			y=best_bbe_2dtensor) # (num_of_rois, 4)
		best_iou_2dtensor = tf.where(
			condition=tf.math.greater(x=iou_2dtensor, y=best_iou_2dtensor),
			x=iou_2dtensor,
			y=best_iou_2dtensor) # (num_of_rois, 1)

	best_cat_2dtensor = tf.where(
		condition=tf.math.greater(x=best_iou_2dtensor, y=iou_threshold_tensor[1]),
		x=best_cat_2dtensor,
		y=num_of_classes-1.0) # (num_of_rois, 1), num_of_classes-1.0 is last class which is background
	cat_1dtensor = tf.reshape(tensor=best_cat_2dtensor, shape=[num_of_rois])
	cat1d = np.array(cat_1dtensor, dtype='int32')
	cat1d[num_of_bboxes:] = num_of_classes # invalidate pad rois
	clz_2dtensor = tf.one_hot(indices=cat1d, depth=num_of_classes) # (num_of_rois, num_of_classes)
	
	bbe_2dtensor = tf.where(
		condition=tf.math.greater(x=best_iou_2dtensor, y=iou_threshold_tensor[1]),
		x=best_bbe_2dtensor,
		y=0.0) # (num_of_rois, 4)
	bbe2d = np.array(bbe_2dtensor, dtype='float32')
	bbe2d[num_of_bboxes:, :] = 0.0 # invalidate pad rois
	bbe_2dtensor = tf.constant(value=bbe2d, dtype='float32') # (num_of_rois, 4)

	clzbbe_2dtensor = tf.concat(values=[clz_2dtensor, bbe_2dtensor], axis=1) # (num_of_rois, num_of_classes+4)

	return clzbbe_2dtensor











