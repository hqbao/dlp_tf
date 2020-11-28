import tensorflow as tf
import numpy as np


def genanchors(isize, ssize, asizes):
	'''
	Arguments
		isize: image size (h, w)
		ssize: feature map size (h, w)
		asizes: list of anchor sizes, (h, w)
	Return 
		abox4d: anchors in images, each anchor has shape (h, w, k, 4). 4 is lenngth of [y1, x1, y2, x2]
	'''

	# scale of feature map for origin image
	xsof = float(ssize[0])/float(isize[0])
	ysof = float(ssize[1])/float(isize[1])

	# all anchors
	# 4 is length of [y1, x1, y2, x2]
	abox4d = np.zeros(shape=(ssize[0], ssize[1], len(asizes), 4), dtype='float32') # channels last

	# iterate over feature map
	for i in range(0, ssize[0]):
		for j in range(0, ssize[1]):
			apoint = [(i + 0.5)/xsof, (j + 0.5)/ysof]

			# iterate over anchor at a point on feature map
			for k in range(0, len(asizes)):
				abox4d[i, j, k] = [
					apoint[0] - asizes[k][0]/2, 
					apoint[1] - asizes[k][1]/2, 
					apoint[0] + asizes[k][0]/2, 
					apoint[1] + asizes[k][1]/2
				]
				
	abox2d = abox4d.reshape((-1, 4))
	return abox2d

def comiou2d(abox_2dtensor, bbox_2dtensor):
	'''
	Compute IoU
	'''

	b1y1 = bbox_2dtensor[:, 0:1]
	b1x1 = bbox_2dtensor[:, 1:2]
	b1y2 = bbox_2dtensor[:, 2:3]
	b1x2 = bbox_2dtensor[:, 3:4]

	b2y1 = abox_2dtensor[:, 0:1]
	b2x1 = abox_2dtensor[:, 1:2]
	b2y2 = abox_2dtensor[:, 2:3]
	b2x2 = abox_2dtensor[:, 3:4]

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

def comiou(bbox, pred_bbox):
	'''
	Compute IoU
	'''

	b1y1 = bbox[0]
	b1x1 = bbox[1]
	b1y2 = bbox[2]
	b1x2 = bbox[3]

	b2y1 = pred_bbox[0]
	b2x1 = pred_bbox[1]
	b2y2 = pred_bbox[2]
	b2x2 = pred_bbox[3]

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

def comloc2d(bbox_2dtensor, abox_2dtensor):
	'''
	Compute bounding box error
	'''

	b1y1 = bbox_2dtensor[:, 0:1]
	b1x1 = bbox_2dtensor[:, 1:2]
	b1y2 = bbox_2dtensor[:, 2:3]
	b1x2 = bbox_2dtensor[:, 3:4]

	b2y1 = abox_2dtensor[:, 0:1]
	b2x1 = abox_2dtensor[:, 1:2]
	b2y2 = abox_2dtensor[:, 2:3]
	b2x2 = abox_2dtensor[:, 3:4]

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

def loc2box2d(box_2dtensor, bbe_2dtensor):
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

def nsm(abox_2dtensor, prediction, nsm_iou_threshold, nsm_score_threshold, nsm_max_output_size, total_classes):
	'''
	'''

	loc_2dtensor = prediction[:, total_classes+1:] # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, 4)
	pbox_2dtensor = loc2box2d(box_2dtensor=abox_2dtensor, bbe_2dtensor=loc_2dtensor) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, 4)

	clz_2dtensor = prediction[:, :total_classes+1] # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, total_classes+1)
	clz_1dtensor = tf.math.argmax(input=clz_2dtensor, axis=-1) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4,)

	cancel = tf.where(
		condition=tf.math.less(x=clz_1dtensor, y=total_classes),
		x=1.0,
		y=0.0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4,)

	score_1dtensor = tf.math.reduce_max(input_tensor=clz_2dtensor, axis=-1) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4,)
	score_1dtensor *= cancel # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4,)

	selected_indices, valid_outputs = tf.image.non_max_suppression_padded(
		boxes=pbox_2dtensor,
		scores=score_1dtensor,
		max_output_size=nsm_max_output_size,
		iou_threshold=nsm_iou_threshold,
		score_threshold=nsm_score_threshold,
		pad_to_max_output_size=True)

	box_2dtensor = tf.gather(params=pbox_2dtensor, indices=selected_indices) # (nsm_max_output_size, 4)
	clz_1dtensor = tf.gather(params=clz_1dtensor, indices=selected_indices) # (nsm_max_output_size,)
	clz_2dtensor = tf.expand_dims(input=clz_1dtensor, axis=1) # (nsm_max_output_size, 1)
	box_2dtensor = tf.cast(x=box_2dtensor, dtype='int32')
	clz_2dtensor = tf.cast(x=clz_2dtensor, dtype='int32')

	boxclz_2dtensor = tf.concat(values=[box_2dtensor, clz_2dtensor], axis=-1)

	return boxclz_2dtensor, valid_outputs

def match(boxes, pboxes, areas=[45**2, 91**2, 181**2, 362**2, 724**2], iou_threshold=0.5):
	'''
	'''

	x_boxes = []
	s_boxes = []
	m_boxes = []
	l_boxes = []
	for y1, x1, y2, x2, _ in boxes:
		a = (y2 - y1)*(x2 - x1)
		if a >= areas[0] and a < areas[1]:
			x_boxes.append([y1, x1, y2, x2])
		elif a >= areas[1] and a < areas[2]:
			s_boxes.append([y1, x1, y2, x2])
		elif a >= areas[2] and a < areas[3]:
			m_boxes.append([y1, x1, y2, x2])
		elif a >= areas[3] and a < areas[4]:
			l_boxes.append([y1, x1, y2, x2])
		else:
			print('Box out of range')

	total_boxes = len(boxes)

	total_x_boxes = len(x_boxes)
	total_s_boxes = len(s_boxes)
	total_m_boxes = len(m_boxes)
	total_l_boxes = len(l_boxes)

	total_pboxes = len(pboxes)

	x_tp = 0
	s_tp = 0
	m_tp = 0
	l_tp = 0

	x_fn = 0
	s_fn = 0
	m_fn = 0
	l_fn = 0

	for i in range(total_boxes):
		intersected = 0
		for j in range(total_pboxes):
			iou = comiou(bbox=boxes[i], pred_bbox=pboxes[j])
			if iou >= iou_threshold:
				intersected = 1
				break

		y1, x1, y2, x2, _ = boxes[i]
		a = (y2 - y1)*(x2 - x1)
		if a < areas[1]:
			x_tp += intersected
			x_fn += int(not intersected)
		elif a >= areas[1] and a < areas[2]:
			s_tp += intersected
			s_fn += int(not intersected)
		elif a >= areas[2] and a < areas[3]:
			m_tp += intersected
			m_fn += int(not intersected)
		elif a >= areas[3]:
			l_tp += intersected
			l_fn += int(not intersected)

	x_fp = 0
	s_fp = 0
	m_fp = 0
	l_fp = 0

	for i in range(total_pboxes):
		not_intersected = 1
		for j in range(total_boxes):
			iou = comiou(bbox=boxes[j], pred_bbox=pboxes[i])
			if iou >= iou_threshold:
				not_intersected = 0
				break

		y1, x1, y2, x2, _ = pboxes[i]
		a = (y2 - y1)*(x2 - x1)
		if a < areas[1]:
			x_fp += not_intersected
		elif a >= areas[1] and a < areas[2]:
			s_fp += not_intersected
		elif a >= areas[2] and a < areas[3]:
			m_fp += not_intersected
		elif a >= areas[3]:
			l_fp += not_intersected

	return total_x_boxes, total_s_boxes, total_m_boxes, total_l_boxes, x_tp, s_tp, m_tp, l_tp, x_fn, s_fn, m_fn, l_fn, x_fp, s_fp, m_fp, l_fp









