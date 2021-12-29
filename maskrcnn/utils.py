import tensorflow as tf
import numpy as np
from math import log, pow 

def genanchors(isize, ssize, asizes):
	'''
	Arguments
		isize: image size (h, w)
		ssize: feature map size (h, w)
		asizes: list of anchor sizes, (h, w)
	Return 
		aboxes: anchors in images, each anchor has shape (h, w, k, 4). 4 is lenngth of [y1, x1, y2, x2]
	'''

	# scale of feature map for origin image
	xsof = float(ssize[0])/float(isize[0])
	ysof = float(ssize[1])/float(isize[1])

	# all anchors
	# 4 is length of [y1, x1, y2, x2]
	aboxes = np.zeros(shape=(ssize[0], ssize[1], len(asizes), 4), dtype='float32') # channels last

	# iterate over feature map
	for i in range(0, ssize[0]):
		for j in range(0, ssize[1]):
			apoint = [(i + 0.5)/xsof, (j + 0.5)/ysof]

			# iterate over anchor at a point on feature map
			for k in range(0, len(asizes)):
				aboxes[i, j, k] = [
					apoint[0] - asizes[k][0]/2, 
					apoint[1] - asizes[k][1]/2, 
					apoint[0] + asizes[k][0]/2, 
					apoint[1] + asizes[k][1]/2
				]

	return aboxes

def combbe4d(bbox_4dtensor, anchor_4dtensor):
	'''
	Compute bounding box error
	'''

	b1y1 = bbox_4dtensor[:, :, :, 0:1]
	b1x1 = bbox_4dtensor[:, :, :, 1:2]
	b1y2 = bbox_4dtensor[:, :, :, 2:3]
	b1x2 = bbox_4dtensor[:, :, :, 3:4]

	b2y1 = anchor_4dtensor[:, :, :, 0:1]
	b2x1 = anchor_4dtensor[:, :, :, 1:2]
	b2y2 = anchor_4dtensor[:, :, :, 2:3]
	b2x2 = anchor_4dtensor[:, :, :, 3:4]

	h = b1y2 - b1y1
	w = b1x2 - b1x1
	y = b1y1 + 0.5*h
	x = b1x1 + 0.5*w

	ha = b2y2 - b2y1
	wa = b2x2 - b2x1
	ya = b2y1 + 0.5*ha
	xa = b2x1 + 0.5*wa

	ty = (y - ya)/ha
	tx = (x - xa)/wa
	th = tf.math.log(h/ha)/tf.math.log(tf.constant(value=2.0, dtype='float32'))
	tw = tf.math.log(w/wa)/tf.math.log(tf.constant(value=2.0, dtype='float32'))

	t = tf.concat(values=[ty, tx, th, tw], axis=3)

	return t

def combbe2d(bbox_2dtensor, roi_2dtensor):
	'''
	Compute bounding box error
	'''

	b1y1 = bbox_2dtensor[:, 0:1]
	b1x1 = bbox_2dtensor[:, 1:2]
	b1y2 = bbox_2dtensor[:, 2:3]
	b1x2 = bbox_2dtensor[:, 3:4]

	b2y1 = roi_2dtensor[:, 0:1]
	b2x1 = roi_2dtensor[:, 1:2]
	b2y2 = roi_2dtensor[:, 2:3]
	b2x2 = roi_2dtensor[:, 3:4]

	h = b1y2 - b1y1
	w = b1x2 - b1x1
	y = b1y1 + 0.5*h
	x = b1x1 + 0.5*w

	ha = b2y2 - b2y1
	wa = b2x2 - b2x1
	ya = b2y1 + 0.5*ha
	xa = b2x1 + 0.5*wa

	ty = (y - ya)/ha
	tx = (x - xa)/wa
	th = tf.math.log(h/ha)/tf.math.log(tf.constant(value=2.0, dtype='float32'))
	tw = tf.math.log(w/wa)/tf.math.log(tf.constant(value=2.0, dtype='float32'))

	t = tf.concat(values=[ty, tx, th, tw], axis=1)

	return t

def comiou4d(bbox_4dtensor, anchor_4dtensor):
	'''
	Compute IoU
	'''

	b1y1 = bbox_4dtensor[:, :, :, 0:1]
	b1x1 = bbox_4dtensor[:, :, :, 1:2]
	b1y2 = bbox_4dtensor[:, :, :, 2:3]
	b1x2 = bbox_4dtensor[:, :, :, 3:4]

	b2y1 = anchor_4dtensor[:, :, :, 0:1]
	b2x1 = anchor_4dtensor[:, :, :, 1:2]
	b2y2 = anchor_4dtensor[:, :, :, 2:3]
	b2x2 = anchor_4dtensor[:, :, :, 3:4]

	b3y1 = tf.math.maximum(x=b1y1, y=b2y1)
	b3y2 = tf.math.minimum(x=b1y2, y=b2y2)
	b3x1 = tf.math.maximum(x=b1x1, y=b2x1)
	b3x2 = tf.math.minimum(x=b1x2, y=b2x2)

	# area of box 1
	s1 = (b1y2 - b1y1) * (b1x2 - b1x1)

	# area of box 2
	s2 = (b2y2 - b2y1) * (b2x2 - b2x1)

	# area of box 3
	s3 = tf.where(
		condition=tf.math.logical_or(
			x=tf.math.less(x=b3y2 - b3y1, y=0.0),
			y=tf.math.less(x=b3x2 - b3x1, y=0.0)),
		x=0.0,
		y=(b3y2 - b3y1) * (b3x2 - b3x1))

	return s3/(s1+s2-s3)

def comiou2d(bbox_2dtensor, roi_2dtensor):
	'''
	Compute IoU
	'''

	b1y1 = bbox_2dtensor[:, 0:1]
	b1x1 = bbox_2dtensor[:, 1:2]
	b1y2 = bbox_2dtensor[:, 2:3]
	b1x2 = bbox_2dtensor[:, 3:4]

	b2y1 = roi_2dtensor[:, 0:1]
	b2x1 = roi_2dtensor[:, 1:2]
	b2y2 = roi_2dtensor[:, 2:3]
	b2x2 = roi_2dtensor[:, 3:4]

	b3y1 = tf.math.maximum(x=b1y1, y=b2y1)
	b3y2 = tf.math.minimum(x=b1y2, y=b2y2)
	b3x1 = tf.math.maximum(x=b1x1, y=b2x1)
	b3x2 = tf.math.minimum(x=b1x2, y=b2x2)

	# area of box 1
	s1 = (b1y2 - b1y1) * (b1x2 - b1x1)

	# area of box 2
	s2 = (b2y2 - b2y1) * (b2x2 - b2x1)

	# area of box 3
	s3 = tf.where(
		condition=tf.math.logical_or(
			x=tf.math.less(x=b3y2 - b3y1, y=0.0),
			y=tf.math.less(x=b3x2 - b3x1, y=0.0)),
		x=0.0,
		y=(b3y2 - b3y1) * (b3x2 - b3x1))

	return s3/(s1+s2-s3)

def comiou(bbox, roi):
	'''
	Compute IoU
	'''

	b1y1 = bbox[0]
	b1x1 = bbox[1]
	b1y2 = bbox[2]
	b1x2 = bbox[3]

	b2y1 = roi[0]
	b2x1 = roi[1]
	b2y2 = roi[2]
	b2x2 = roi[3]

	b3y1 = max(b1y1, b2y1)
	b3y2 = min(b1y2, b2y2)
	b3x1 = max(b1x1, b2x1)
	b3x2 = min(b1x2, b2x2)

	# area of box 1
	s1 = (b1y2 - b1y1) * (b1x2 - b1x1)

	# area of box 2
	s2 = (b2y2 - b2y1) * (b2x2 - b2x1)

	if b3y2 < b3y1:
		return 0

	if b3x2 < b3x1:
		return 0

	s3 = (b3y2 - b3y1) * (b3x2 - b3x1)

	return s3/(s1+s2-s3)

def bbe2box4d(box_4dtensor, bbe_4dtensor):
	'''
	Arguments
		box_4dtensor: (h, w, k, 4)
		bbe_4dtensor: (h, w, k, 4)
	Return
		tensor
	'''

	ya1 =  box_4dtensor[:, :, :, 0:1]
	xa1 =  box_4dtensor[:, :, :, 1:2]
	ya2 =  box_4dtensor[:, :, :, 2:3]
	xa2 =  box_4dtensor[:, :, :, 3:4]

	ha = ya2 - ya1
	wa = xa2 - xa1
	ya = ya1 + 0.5*ha
	xa = xa1 + 0.5*wa

	ty = bbe_4dtensor[:, :, :, 0:1]
	tx = bbe_4dtensor[:, :, :, 1:2]
	th = bbe_4dtensor[:, :, :, 2:3]
	tw = bbe_4dtensor[:, :, :, 3:4]

	# Clip ty, tx, th, tw
	# ty = tf.clip_by_value(t=ty, clip_value_min=-1024, clip_value_max=1024)
	# tx = tf.clip_by_value(t=tx, clip_value_min=-1024, clip_value_max=1024)
	# th = tf.clip_by_value(t=th, clip_value_min=-10, clip_value_max=10)
	# tw = tf.clip_by_value(t=tw, clip_value_min=-10, clip_value_max=10)

	y = ty*ha + ya
	x = tx*wa + xa
	h = tf.math.pow(2.0, th)*ha
	w = tf.math.pow(2.0, tw)*wa

	y1 = y - 0.5*h
	x1 = x - 0.5*w
	y2 = y + 0.5*h
	x2 = x + 0.5*w

	t = tf.concat(values=[y1, x1, y2, x2], axis=3)

	return t

def bbe2box2d(box_2dtensor, bbe_2dtensor):
	'''
	Arguments
		box_4dtensor: (num_of_boxes, 4)
		bbe_2dtensor: (num_of_boxes, 4)
	Return
		tensor
	'''

	ya1 =  box_2dtensor[:, 0:1]
	xa1 =  box_2dtensor[:, 1:2]
	ya2 =  box_2dtensor[:, 2:3]
	xa2 =  box_2dtensor[:, 3:4]

	ha = ya2 - ya1
	wa = xa2 - xa1
	ya = ya1 + 0.5*ha
	xa = xa1 + 0.5*wa

	ty = bbe_2dtensor[:, 0:1]
	tx = bbe_2dtensor[:, 1:2]
	th = bbe_2dtensor[:, 2:3]
	tw = bbe_2dtensor[:, 3:4]

	# Clip ty, tx, th, tw
	# ty = tf.clip_by_value(t=ty, clip_value_min=-1024, clip_value_max=1024)
	# tx = tf.clip_by_value(t=tx, clip_value_min=-1024, clip_value_max=1024)
	# th = tf.clip_by_value(t=th, clip_value_min=-10, clip_value_max=10)
	# tw = tf.clip_by_value(t=tw, clip_value_min=-10, clip_value_max=10)

	y = ty*ha + ya
	x = tx*wa + xa
	h = tf.math.pow(2.0, th)*ha
	w = tf.math.pow(2.0, tw)*wa

	y1 = y - 0.5*h
	x1 = x - 0.5*w
	y2 = y + 0.5*h
	x2 = x + 0.5*w

	t = tf.concat(values=[y1, x1, y2, x2], axis=1)

	return t

def normbox2d(box_2dtensor, isize):
	'''
	Arguments
		box_2dtensor: (n, 4)
		isize: image size
	Return 
		norm_box_2dtensor
	'''

	# box
	y1 = box_2dtensor[:, 0:1]
	x1 = box_2dtensor[:, 1:2]
	y2 = box_2dtensor[:, 2:3]
	x2 = box_2dtensor[:, 3:4]

	# normalize
	y1 /= isize[0]
	x1 /= isize[1]
	y2 /= isize[0]
	x2 /= isize[1]

	y1 = tf.math.maximum(y1, 0)
	x1 = tf.math.maximum(x1, 0)
	y2 = tf.math.minimum(y2, 1)
	x2 = tf.math.minimum(x2, 1)

	norm_box_2dtensor = tf.concat(values=[y1, x1, y2, x2], axis=1)

	return norm_box_2dtensor

def unnormbox2d(box_2dtensor, isize):
	'''
	Arguments
		box_2dtensor: (n, 4)
		isize: image size

	Return 
		normalized box_2dtensor
	'''

	# box
	y1 = box_2dtensor[:, 0:1]
	x1 = box_2dtensor[:, 1:2]
	y2 = box_2dtensor[:, 2:3]
	x2 = box_2dtensor[:, 3:4]

	# normalize
	y1 *= isize[0]
	x1 *= isize[1]
	y2 *= isize[0]
	x2 *= isize[1]

	y1 = tf.math.maximum(y1, 0)
	x1 = tf.math.maximum(x1, 0)
	y2 = tf.math.minimum(y2, isize[0])
	x2 = tf.math.minimum(x2, isize[1])

	t = tf.concat(values=[y1, x1, y2, x2], axis=1)

	return t

def box2frame(box, apoint=[0.5, 0.5]):
	'''
	Convert [y1, x1, y2, x2] to [x, y, w, h]
	'''

	return [
		(box[1] + apoint[1]*(box[3]-box[1])),
		(box[0] + apoint[0]*(box[2]-box[0])),
		(box[3] - box[1]),
		(box[2] - box[0])
	]

def pad_roi2d(roi2d, max_num_of_rois, pad_roi=[0, 0, 1, 1]):
	'''
	Arguments
		roi2d:
		max_num_of_rois
	Return 
		rois: normalized, applied scale
	'''

	padded_roi2d = np.zeros((max_num_of_rois, 4), dtype='float32')

	for i in range(max_num_of_rois):
		if i < roi2d.shape[0]:
			padded_roi2d[i] = roi2d[i]
		else:
			padded_roi2d[i] = np.array(pad_roi, dtype='float32')

	return padded_roi2d

def nsm(anchor_4dtensor, clzbbe_3dtensor, max_num_of_rois, nsm_iou_threshold, nsm_score_threshold, ishape):
	'''
	Non max suppression in case non FPN
	'''

	h = int(clzbbe_3dtensor.shape[0])
	w = int(clzbbe_3dtensor.shape[1])
	k = int(clzbbe_3dtensor.shape[2]/6)

	pbox_2dtensor = tf.constant(value=[[0, 0, ishape[0], ishape[1]]], dtype='float32') # pad roi
	score_1dtensor = tf.constant(value=[0], dtype='float32')

	clz_3dtensor = clzbbe_3dtensor[:, :, :2*k] # (h, w, 2k)
	bbe_3dtensor = clzbbe_3dtensor[:, :, 2*k:] # (h, w, 4k)
	
	# bbe shape (h, w, 4k) -> (h, w, k, 4)
	bbe_4dtensor = tf.reshape(tensor=bbe_3dtensor, shape=[h, w, k, 4])

	# boxes for non max suppression
	pbox_4dtensor = bbe2box4d(box_4dtensor=anchor_4dtensor, bbe_4dtensor=bbe_4dtensor)
	_pbox_2dtensor = tf.reshape(tensor=pbox_4dtensor, shape=[h*w*k, 4])

	# score for non max suppression
	fg_score_3dtensor = clz_3dtensor[:, :, ::2] # foreground probs
	bg_score_3dtensor = clz_3dtensor[:, :, 1::2] # background probs
	score_3dtensor = tf.where(
		condition=tf.math.greater(x=fg_score_3dtensor, y=bg_score_3dtensor),
		x=fg_score_3dtensor,
		y=0.0)
	_score_1dtensor = tf.reshape(tensor=score_3dtensor, shape=[h*w*k])

	pbox_2dtensor = tf.concat(values=[pbox_2dtensor, _pbox_2dtensor], axis=0)
	score_1dtensor = tf.concat(values=[score_1dtensor, _score_1dtensor], axis=0)

	# non max suppression
	selected_indices, _ = tf.image.non_max_suppression_padded(
		boxes=pbox_2dtensor,
		scores=score_1dtensor,
		max_output_size=max_num_of_rois,
		iou_threshold=nsm_iou_threshold,
		score_threshold=nsm_score_threshold,
		pad_to_max_output_size=True)
	roi_2dtensor = tf.gather(pbox_2dtensor, selected_indices) # (max_num_of_rois, 4)

	return roi_2dtensor

def pnsm(anchor_4dtensors, clzbbe_3dtensors, max_num_of_rois, nsm_iou_threshold, nsm_score_threshold, ishape):
	'''
	Non max suppression in case FPN
	'''

	pbox_2dtensor = tf.constant(value=[[0, 0, ishape[0], ishape[1]]], dtype='float32')
	score_1dtensor = tf.constant(value=[0], dtype='float32')

	for lvl in range(4):
		clzbbe_3dtensor = clzbbe_3dtensors[lvl] # (h, w, 6k)

		h = int(clzbbe_3dtensor.shape[0])
		w = int(clzbbe_3dtensor.shape[1])
		k = int(clzbbe_3dtensor.shape[2]/6)

		clz_tensor = clzbbe_3dtensor[:, :, :2*k] # (h, w, 2k)
		bbe_tensor = clzbbe_3dtensor[:, :, 2*k:] # (h, w, 4k)
		
		anchor_4dtensor = anchor_4dtensors[lvl] # (h, w, k, 4)

		# bbe shape (h, w, 4k) -> (h, w, k, 4)
		bbe_4dtensor = tf.reshape(tensor=bbe_tensor, shape=[h, w, k, 4])

		# boxes for non max suppression
		pbox_4dtensor = bbe2box4d(box_4dtensor=anchor_4dtensor, bbe_4dtensor=bbe_4dtensor)
		_pbox_2dtensor = tf.reshape(tensor=pbox_4dtensor, shape=[h*w*k, 4])

		# score for non max suppression
		fg_score_3dtensor = clz_tensor[:, :, ::2] # foreground probs
		bg_score_3dtensor = clz_tensor[:, :, 1::2] # background probs
		_score_1dtensor = tf.where(
			condition=tf.math.greater(x=fg_score_3dtensor, y=bg_score_3dtensor),
			x=fg_score_3dtensor,
			y=0.0)
		_score_1dtensor = tf.reshape(tensor=_score_1dtensor, shape=[h*w*k])

		pbox_2dtensor = tf.concat(values=[pbox_2dtensor, _pbox_2dtensor], axis=0)
		score_1dtensor = tf.concat(values=[score_1dtensor, _score_1dtensor], axis=0)

	# non max suppression
	selected_indices, _ = tf.image.non_max_suppression_padded(
		boxes=pbox_2dtensor,
		scores=score_1dtensor,
		max_output_size=max_num_of_rois,
		iou_threshold=nsm_iou_threshold,
		score_threshold=nsm_score_threshold,
		pad_to_max_output_size=True) 
	roi_2dtensor = tf.gather(pbox_2dtensor, selected_indices) # (max_num_of_rois, 4)

	return roi_2dtensor







	