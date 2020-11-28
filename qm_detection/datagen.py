import tensorflow as tf
import numpy as np
from skimage import io, transform
from random import randint
from scipy.stats import multivariate_normal
from utils import comiou2d, comloc2d
from datetime import datetime


def randcrop(image, bboxes, ishape):
	'''
	'''

	bbox_len = len(bboxes)
	bbox_idx = randint(0, bbox_len-1)
	y1, x1, y2, x2, _ = list(map(int, bboxes[bbox_idx]))
	h = y2 - y1
	w = x2 - x1

	crop_y1 = y1 - randint(0, max(int(ishape[0]-h-1), 0))
	crop_y2 = crop_y1 + ishape[0]
	if crop_y1 < 0:
		crop_y1 = 0
		crop_y2 = ishape[0]
	if crop_y2 > image.shape[0]:
		crop_y1 = image.shape[0] - ishape[0]
		crop_y2 = image.shape[0]

	crop_x1 = x1 - randint(0, max(int(ishape[1]-w-1), 0))
	crop_x2 = crop_x1 + ishape[1]
	if crop_x1 < 0:
		crop_x1 = 0
		crop_x2 = ishape[1]
	if crop_x2 > image.shape[1]:
		crop_x1 = image.shape[1] - ishape[1]
		crop_x2 = image.shape[1]

	crop_y1, crop_x1, crop_y2, crop_x2 = int(crop_y1), int(crop_x1), int(crop_y2), int(crop_x2)
	cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2, :]

	cropped_image = cropped_image/np.max(cropped_image)
	cropped_image *= 255
	cropped_image = np.array(cropped_image, dtype='int32')

	remain_bboxes = []
	for i in range(bbox_len):
		y1, x1, y2, x2, cat = bboxes[i]
		h = y2 - y1
		w = x2 - x1
		if crop_y1 - y1 > 0.5*h:
			continue
		if crop_x1 - x1 > 0.5*w:
			continue
		if y2 - crop_y2 > 0.5*h:
			continue
		if x2 - crop_x2 > 0.5*w:
			continue

		remain_bboxes.append([y1-crop_y1, x1-crop_x1, y2-crop_y1, x2-crop_x1, cat])	

	return [cropped_image, remain_bboxes]

def randaug(image, bboxes, scale, ishape, mode):
	'''
	'''

	# Zoom
	bbox_len = len(bboxes)
	zoom_image = transform.resize(image=image, output_shape=[int(scale*image.shape[0]), int(scale*image.shape[1])])
	zoom_image = zoom_image/np.max(zoom_image)
	zoom_image = zoom_image*255
	zoom_bboxes = []
	for i in range(bbox_len):
		y1, x1, y2, x2, cat = bboxes[i]
		zoom_bboxes.append([scale*y1, scale*x1, scale*y2, scale*x2, cat])

	# Crop
	image, remain_bboxes = randcrop(image=zoom_image, bboxes=zoom_bboxes, ishape=ishape)

	if mode == 'train':
		# Flip
		if randint(0, 1) == 0:
			image = np.fliplr(image)
			for i in range(len(remain_bboxes)):
				_, x1, _, x2, _ = remain_bboxes[i]
				remain_bboxes[i][1] = ishape[1] - x2
				remain_bboxes[i][3] = ishape[1] - x1

		# Change color unifiedly
		if randint(0, 1) == 0 or True:
			image[:, :, 0] += randint(0, 255)
			image[:, :, 1] += randint(0, 255)
			image[:, :, 2] += randint(0, 255)
			image = image/np.max(image)
			image = image*255
			image = np.array(image, dtype='int32')

		# Grey down
		if randint(0, 1) == 0:
			down = randint(50, 100)/100
			image = image*down
			image = np.array(image, dtype='int32')

		# Contrast
		if randint(0, 1) == 0 or True:
			image = image - 127
			image = image*(randint(100, 150)/100)
			image = image + 127
			image = np.clip(image, 0, 255)
			image = np.array(image, dtype='int32')

		# Saturate
		if randint(0, 1) == 0 or True:
			image = np.clip(image, randint(0, 32), randint(224, 255))
			image = image/np.max(image)
			image = image*255
			image = np.array(image, dtype='int32')

		# Add heatmap
		if randint(0, 1) == 0 or True:
			pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
			center_y = randint(0, ishape[0])
			center_x = randint(0, ishape[1])
			rv = multivariate_normal(mean=[center_y, center_x], cov=randint(0, 5000), allow_singular=True)
			heatmap2d = rv.pdf(pos)
			heatmap2d /= np.max(heatmap2d)
			heatmap2d = heatmap2d*randint(0, 128)
			heatmap2d = np.array(heatmap2d, dtype='int32')
			image[:, :, randint(0, 2)] += heatmap2d
			image = image/np.max(image)
			image = image*255
			image = np.array(image, dtype='int32')

		# Add heatmap
		if randint(0, 1) == 0 or True:
			pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
			center_y = randint(0, ishape[0])
			center_x = randint(0, ishape[1])
			rv = multivariate_normal(mean=[center_y, center_x], cov=randint(0, 5000), allow_singular=True)
			heatmap2d = rv.pdf(pos)
			heatmap2d /= np.max(heatmap2d)
			heatmap2d = heatmap2d*randint(0, 128)
			heatmap2d = np.array(heatmap2d, dtype='int32')
			image[:, :, randint(0, 2)] += heatmap2d
			image = image/np.max(image)
			image = image*255
			image = np.array(image, dtype='int32')

		# Add heatmap
		if randint(0, 1) == 0 or True:
			pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
			center_y = randint(0, ishape[0])
			center_x = randint(0, ishape[1])
			rv = multivariate_normal(mean=[center_y, center_x], cov=randint(0, 5000), allow_singular=True)
			heatmap2d = rv.pdf(pos)
			heatmap2d /= np.max(heatmap2d)
			heatmap2d = heatmap2d*randint(0, 128)
			heatmap2d = np.array(heatmap2d, dtype='int32')
			image[:, :, randint(0, 2)] += heatmap2d
			image = image/np.max(image)
			image = image*255
			image = np.array(image, dtype='int32')

		# Lose feature
		if randint(0, 1) == 0 or True:
			scale = randint(50, 100)/100
			image = transform.resize(image=image, output_shape=[int(scale*ishape[0]), int(scale*ishape[1])])
			image = transform.resize(image=image, output_shape=[int(ishape[0]), int(ishape[1])])
			image = image/np.max(image)
			image = image*255
			image = np.array(image, dtype='int32')

	return [image, remain_bboxes]

def genxy(anno_file_path, image_dir, ishape, abox_2dtensor, iou_thresholds, total_classes, anchor_sampling, mode):
	'''
	Arguments
		anno_file_path:
		image_dir:
		ishape:
		abox_2dtensor:
	Return 
		tensor: (h*w*k, 1+4)
	'''

	anno_file = open(anno_file_path, 'r')

	lines = anno_file.readlines()
	total_lines = len(lines)
	# print('\nTotal lines: {}'.format(total_lines))
	np.random.shuffle(lines)

	for line_idx in range(total_lines):
		line = lines[line_idx]
		anno = line[:-1].split(' ')
		image_id = anno[0]
		bboxes = anno[1:]
		bboxes = [list(map(float, bboxes[i:i+5])) for i in range(0, len(bboxes), 5)]

		image_file_path = image_dir + '/' + image_id + '.jpg'
		image = io.imread(image_file_path)
		image, bboxes = randaug(image=image, bboxes=bboxes, scale=randint(90, 110)/100, ishape=ishape, mode=mode)

		bbox_2dtensor = tf.constant(value=bboxes, dtype='float32') # (total_bboxes, 4)
		bbox_3dtensor = tf.repeat(input=[bbox_2dtensor], repeats=[abox_2dtensor.shape[0]], axis=0) # (h*w*k, total_bboxes, 4)
		bbox_3dtensor = tf.transpose(a=bbox_3dtensor, perm=[1, 0, 2]) # (total_bboxes, h*w*k, 4)

		iou3d = np.zeros((bbox_2dtensor.shape[0], abox_2dtensor.shape[0], 1), dtype='float32') # (total_bboxes, h*w*k, 1)
		for i in range(bbox_3dtensor.shape[0]):
			iou_2dtensor = comiou2d(abox_2dtensor=abox_2dtensor, bbox_2dtensor=bbox_3dtensor[i]) # (h*w*k, 1)
			iou3d[i] = iou_2dtensor

		iou_3dtensor = tf.constant(value=iou3d, dtype='float32') # (total_bboxes, h*w*k, 1)
		iou_2dtensor = tf.squeeze(input=iou_3dtensor, axis=-1) # (total_bboxes, h*w*k)
		iou_2dtensor = tf.transpose(a=iou_2dtensor, perm=[1, 0]) # (h*w*k, total_bboxes)
		iou_1dtensor = tf.math.reduce_max(input_tensor=iou_2dtensor, axis=-1) # (h*w*k,)
		bbox_indice_1dtensor = tf.math.argmax(input=iou_2dtensor, axis=-1) # (h*w*k,)

		# Assign positives, neutral, negatives, zero negatives
		bbox_indice_1dtensor = tf.where(
			condition=tf.math.greater_equal(x=iou_1dtensor, y=iou_thresholds[1]),
			x=bbox_indice_1dtensor,
			y=len(bboxes)+1) # (h*w*k,)
		bbox_indice_1dtensor = tf.where(
			condition=tf.math.logical_and(
				x=tf.math.less(x=iou_1dtensor, y=iou_thresholds[1]),
				y=tf.math.greater(x=iou_1dtensor, y=iou_thresholds[0])),
			x=len(bboxes)+2,
			y=bbox_indice_1dtensor) # (h*w*k,)
		bbox_indice_1dtensor = tf.where(
			condition=tf.math.logical_and(
				x=tf.math.less_equal(x=iou_1dtensor, y=iou_thresholds[0]),
				y=tf.math.greater(x=iou_1dtensor, y=0)),
			x=len(bboxes),
			y=bbox_indice_1dtensor) # (h*w*k,)

		# Sample anchors, pad or remove
		bbox_indice1d = np.array(bbox_indice_1dtensor)
		pos_indices, = np.where(bbox_indice1d < len(bboxes))
		neg_indices, = np.where(bbox_indice1d == len(bboxes))
		zero_neg_indices, = np.where(bbox_indice1d == len(bboxes)+1)

		total_fg = anchor_sampling//2
		total_bg = anchor_sampling - total_fg
		total_pos = len(pos_indices)
		total_neg = len(neg_indices)
		total_zero_neg = len(zero_neg_indices)
		
		if total_pos == 0:
			print('!')
			continue

		if total_pos > total_fg:
			np.random.shuffle(pos_indices)
			remove_pos_indices = pos_indices[total_fg:]
			for i in remove_pos_indices:
				bbox_indice1d[i] = len(bboxes)+2

		else:
			total_fg = total_pos
			total_bg = anchor_sampling - total_pos

		if total_neg > total_bg//2:
			total_selected_zero_neg = min(total_bg//2, total_zero_neg)
			total_selected_neg = min(total_bg - total_selected_zero_neg, total_neg)

		if total_zero_neg > total_bg//2:
			total_selected_neg = min(total_bg//2, total_neg)
			total_selected_zero_neg = min(total_bg - total_selected_neg, total_zero_neg)

		if total_selected_neg + total_selected_zero_neg < total_bg:
			print('?')
			continue
			
		np.random.shuffle(neg_indices)
		bbox_indice1d[neg_indices[total_selected_neg:]] = len(bboxes)+2

		np.random.shuffle(zero_neg_indices)
		bbox_indice1d[zero_neg_indices[:total_selected_zero_neg]] = len(bboxes)

		# Compute loc error
		bboxes.append([0, 0, 0, 0, total_classes])
		bboxes.append([0, 0, 0, 0, total_classes+1])
		bboxes.append([0, 0, 0, 0, total_classes+1])
		matching_2dtensor = tf.gather(params=bboxes, indices=bbox_indice1d) # (h*w*k, 5)
		matching_bbox_2dtensor = matching_2dtensor[:, :4] # (h*w*k, 4)
		matching_cat_1dtensor = matching_2dtensor[:, 4] # (h*w*k,)
		clz_2dtensor = tf.one_hot(indices=tf.cast(x=matching_cat_1dtensor, dtype='int64'), depth=total_classes+1, axis=-1) # (h*w*k, total_classes+1)
		loc_2dtensor = comloc2d(bbox_2dtensor=matching_bbox_2dtensor, abox_2dtensor=abox_2dtensor) # (h*w*k, 4)
		
		batchy_2dtensor = tf.concat(values=[clz_2dtensor, loc_2dtensor], axis=-1) # (h*w*k, total_classes+1+4)
		batchx_4dtensor = tf.constant(value=[image], dtype='int32')

		yield batchx_4dtensor, batchy_2dtensor, bboxes[:-3], image_id

def genx(anno_file_path, image_dir, ishape, mode):
	'''
	Arguments
		anno_file_path:
		image_dir:
		ishape:
	Return 
		tensor: (h*w*k, 1+4)
	'''

	anno_file = open(anno_file_path, 'r')

	lines = anno_file.readlines()
	total_lines = len(lines)
	# print('\nTotal lines: {}'.format(total_lines))
	np.random.shuffle(lines)

	for line_idx in range(total_lines):
		line = lines[line_idx]
		anno = line[:-1].split(' ')
		image_id = anno[0]
		bboxes = anno[1:]
		bboxes = [list(map(float, bboxes[i:i+5])) for i in range(0, len(bboxes), 5)]

		image_file_path = image_dir + '/' + image_id + '.jpg'
		image = io.imread(image_file_path)
		image, bboxes = randaug(image=image, bboxes=bboxes, scale=1.0, ishape=ishape, mode=mode)
		bbox2d = np.array(bboxes, dtype='int32')

		batchx_4dtensor = tf.constant(value=[image], dtype='int32')
		batchy_2dtensor = tf.constant(value=bbox2d, dtype='int32')

		yield batchx_4dtensor, batchy_2dtensor, image_id












