import tensorflow as tf
import numpy as np
from skimage import io, transform
from random import randint
from scipy.stats import multivariate_normal
from utils import comiou2d, comloc2d


def contrast(image):
	'''
	'''

	image = image - 127
	image = image*(randint(100, 150)/100)
	image = image + 127
	image = np.clip(image, 0, 255)
	return image

def saturate(image):
	'''
	'''

	image = np.clip(image, randint(0, 32), randint(224, 255))
	image = image/np.max(image)
	image = image*255
	return image

def imbalance(image):
	'''
	'''

	image[:, :, 0] += randint(0, 128)
	image[:, :, 1] += randint(0, 128)
	image[:, :, 2] += randint(0, 128)
	image = image/np.max(image)
	image = image*255
	return image

def shine(image, ishape):
	'''
	'''

	pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
	center_y = randint(0, ishape[0])
	center_x = randint(0, ishape[1])
	rv = multivariate_normal(mean=[center_y, center_x], cov=randint(0, 20000), allow_singular=True)
	heatmap2d = rv.pdf(pos)
	heatmap2d /= np.max(heatmap2d)
	heatmap2d = heatmap2d*randint(0, 128)
	indices = [0, 1, 2]
	np.random.shuffle(indices)
	indices = indices[:2]
	image[:, :, indices[0]] += heatmap2d
	image[:, :, indices[1]] += heatmap2d
	image = image/np.max(image)
	image = image*255
	return image

def blur(image, ishape):
	'''
	'''

	scale = randint(50, 100)/100
	image = transform.resize(image=image, output_shape=[int(scale*ishape[0]), int(scale*ishape[1])])
	image = transform.resize(image=image, output_shape=[int(ishape[0]), int(ishape[1])])
	image = image/np.max(image)
	image = image*255
	return image

def augcolor(image, ishape):
	'''
	'''

	# Change color unifiedly
	image = imbalance(image=image)

	# Contrast
	image = contrast(image=image)

	# Saturate
	image = saturate(image=image)

	# Shine
	image = shine(image=image, ishape=ishape)

	# Lose feature
	image = blur(image=image, ishape=ishape)

	# Grey down
	image = np.clip(image - randint(0, 64), 0, 255)

	# Bright up
	image = np.clip(image + randint(0, 64), 0, 255)

	return image

def flip(image, bboxes, ishape, mode):
	'''
	'''

	if mode == 1:
		image = np.fliplr(image)
		for i in range(len(bboxes)):
			_, x1, _, x2 = bboxes[i]
			bboxes[i][1] = ishape[1] - x2
			bboxes[i][3] = ishape[1] - x1

	elif mode == 2:
		image = np.flipud(image)
		for i in range(len(bboxes)):
			y1, _, y2, _ = bboxes[i]
			bboxes[i][0] = ishape[0] - y2
			bboxes[i][2] = ishape[0] - y1

	return image, bboxes

def rotate90(image, bboxes, ishape):
	'''
	'''

	image = np.transpose(image, axes=[1, 0, 2])
	for i in range(len(bboxes)):
		y1, x1, y2, x2 = bboxes[i]
		bboxes[i] = [x1, y1, x2, y2]

	return image, bboxes

def zoom(image, bboxes, scale):
	'''
	'''

	bbox_len = len(bboxes)
	zoom_image = transform.resize(image=image, output_shape=[int(scale*image.shape[0]), int(scale*image.shape[1])])
	zoom_image = zoom_image/np.max(zoom_image)
	zoom_image = zoom_image*255
	zoom_bboxes = []
	for i in range(bbox_len):
		y1, x1, y2, x2 = bboxes[i]
		zoom_bboxes.append([scale*y1, scale*x1, scale*y2, scale*x2])

	return zoom_image, zoom_bboxes

def randcrop(image, bboxes, ishape):
	'''
	'''

	bbox_len = len(bboxes)
	bbox_idx = randint(0, bbox_len-1)
	y1, x1, y2, x2 = list(map(int, bboxes[bbox_idx]))
	h = y2 - y1
	w = x2 - x1

	crop_y1 = y1 - randint(0, max(int(ishape[0]-h), 0))
	crop_y2 = crop_y1 + ishape[0]
	if crop_y1 < 0:
		crop_y1 = 0
		crop_y2 = ishape[0]
	if crop_y2 > image.shape[0]:
		crop_y1 = image.shape[0] - ishape[0]
		crop_y2 = image.shape[0]

	crop_x1 = x1 - randint(0, max(int(ishape[1]-w), 0))
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

	remain_bboxes = []
	for i in range(bbox_len):
		y1, x1, y2, x2 = bboxes[i]
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

		remain_bboxes.append([y1-crop_y1, x1-crop_x1, y2-crop_y1, x2-crop_x1])

	return [cropped_image, remain_bboxes]

def place(image, position, placed_image, bboxes):
	'''
	'''

	ishape = image.shape
	pshape = placed_image.shape
	y, x = position
	top, left, bottom, right = y, x, ishape[0]-y, ishape[1]-x
	py, px = pshape[0]//2, pshape[1]//2
	ptop, pleft, pbottom, pright = pshape[0]//2, pshape[1]//2, pshape[0]-pshape[0]//2, pshape[1]-pshape[1]//2

	pcrop_y1 = max(0, py-top)
	pcrop_x1 = max(0, px-left)
	pcrop_y2 = min(pshape[0], py+bottom)
	pcrop_x2 = min(pshape[1], px+right)

	crop_y1 = max(0, y-ptop)
	crop_x1 = max(0, x-pleft)
	crop_y2 = min(ishape[0], y+pbottom)
	crop_x2 = min(ishape[1], x+pright)

	placed_image = placed_image[pcrop_y1:pcrop_y2, pcrop_x1:pcrop_x2, :]

	bbox_len = len(bboxes)
	remain_bboxes = []
	for i in range(bbox_len):
		y1, x1, y2, x2 = bboxes[i]
		h = y2 - y1
		w = x2 - x1
		if pcrop_y1 - y1 > 0.5*h:
			continue
		if pcrop_x1 - x1 > 0.5*w:
			continue
		if y2 - pcrop_y2 > 0.5*h:
			continue
		if x2 - pcrop_x2 > 0.5*w:
			continue

		remain_bboxes.append([
			y1-pcrop_y1+crop_y1, x1-pcrop_x1+crop_x1, 
			y2-pcrop_y1+crop_y1, x2-pcrop_x1+crop_x1,
		])

	image[crop_y1:crop_y2, crop_x1:crop_x2, :] += placed_image

	return image, remain_bboxes

def nest(images, anno, ishape):
	'''
	'''

	scales = np.array([2.0, 1.0, 0.5, 0.25])
	positions = np.array([
		[256, 256],
		[256, 256],
		[256, 256],
		[randint(128, 384), randint(128, 384)],
	])
	rindices = [0, 1, 2, 3]
	np.random.shuffle(rindices)
	rindices = rindices[:2]

	scales = scales[rindices]
	positions = positions[rindices]

	image = np.zeros((ishape[0], ishape[1], 3), dtype='float32')
	bboxes = []

	for i in range(len(positions)):
		zoom_image, zoom_bboxes = zoom(image=images[i], bboxes=anno[i], scale=scales[i])
		if scales[i] > 1:
			crop_image, crop_bboxes = randcrop(image=zoom_image, bboxes=zoom_bboxes, ishape=ishape)
		else:
			crop_image, crop_bboxes = randcrop(image=zoom_image, bboxes=zoom_bboxes, ishape=[scales[i]*ishape[0], scales[i]*ishape[1]])
		flip_image, flip_bboxes = flip(image=crop_image, bboxes=crop_bboxes, ishape=crop_image.shape[:2], mode=randint(0, 2))
		image, _bboxes = place(image=image, position=positions[i], placed_image=flip_image, bboxes=flip_bboxes)
		bboxes += _bboxes

	image = image/np.max(image)
	image = image*255
	
	return image, bboxes

def patch(images, anno, ishape):
	'''
	'''

	scales = [1.7, 1.0, 0.5, 0.25]
	np.random.shuffle(scales)
	positions = [
		[128, 128], [128, 384],
		[384, 128], [384, 384],
	]
	image = np.zeros((ishape[0], ishape[1], 3), dtype='float32')
	bboxes = []

	for i in range(len(positions)):
		zoom_image, zoom_bboxes = zoom(image=images[i], bboxes=anno[i], scale=scales[i])
		crop_image, crop_bboxes = randcrop(image=zoom_image, bboxes=zoom_bboxes, ishape=[ishape[0]//2, ishape[1]//2])
		flip_image, flip_bboxes = flip(image=crop_image, bboxes=crop_bboxes, ishape=crop_image.shape[:2], mode=randint(0, 2))
		image, _bboxes = place(image=image, position=positions[i], placed_image=flip_image, bboxes=flip_bboxes)
		bboxes += _bboxes

	image = image/np.max(image)
	image = image*255
	
	return image, bboxes

def create_image(images, anno, ishape, mode):
	'''
	'''

	if mode == 0:
		image, bboxes = patch(images, anno, ishape)
	elif mode == 1:
		image, bboxes = nest(images, anno, ishape)
	elif mode == 2:
		image, bboxes = patch(images, anno, ishape)
		image = augcolor(image=image, ishape=ishape)
	elif mode == 3:
		image, bboxes = nest(images, anno, ishape)
		image = augcolor(image=image, ishape=ishape)

	if randint(0, 1) == 0:
		image, bboxes = rotate90(image=image, bboxes=bboxes, ishape=ishape)

	if randint(0, 1) == 1:
		image = np.mean(image, axis=-1, keepdims=True)
		image = np.concatenate([image, image, image], axis=-1)

	return image, bboxes

def parse_line(line):
	'''
	'''

	anno = line[:-1].split(' ')
	image_id = anno[0]
	bboxes = anno[1:]
	bboxes = [list(map(float, bboxes[i:i+4])) for i in range(0, len(bboxes), 5)]
	return image_id, bboxes

def gentiery(bboxes, abox_2dtensor, iou_thresholds, total_classes, anchor_sampling):
	'''
	'''

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
	anchor_iou_1dtensor = tf.math.reduce_max(input_tensor=iou_2dtensor, axis=-1) # (h*w*k,)
	anchor_type_1dtensor = tf.math.argmax(input=iou_2dtensor, axis=-1) # (h*w*k,)

	# Assign positives, neutral, negatives, zero negatives
	anchor_type_1dtensor = tf.where(
		condition=tf.math.greater_equal(x=anchor_iou_1dtensor, y=iou_thresholds[1]),
		x=anchor_type_1dtensor,
		y=len(bboxes)+1) # (h*w*k,)
	anchor_type_1dtensor = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.less(x=anchor_iou_1dtensor, y=iou_thresholds[1]),
			y=tf.math.greater(x=anchor_iou_1dtensor, y=iou_thresholds[0])),
		x=len(bboxes)+2,
		y=anchor_type_1dtensor) # (h*w*k,)
	anchor_type_1dtensor = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.less_equal(x=anchor_iou_1dtensor, y=iou_thresholds[0]),
			y=tf.math.greater(x=anchor_iou_1dtensor, y=0)),
		x=len(bboxes),
		y=anchor_type_1dtensor) # (h*w*k,)

	# Sample anchors, pad or remove
	anchor_type1d = np.array(anchor_type_1dtensor)
	pos_indices, = np.where(anchor_type1d < len(bboxes))
	neg_indices, = np.where(anchor_type1d == len(bboxes))
	zero_neg_indices, = np.where(anchor_type1d == len(bboxes)+1)

	total_fg = anchor_sampling//2
	total_bg = anchor_sampling - total_fg
	total_pos = len(pos_indices)
	total_neg = len(neg_indices)
	total_zero_neg = len(zero_neg_indices)
	
	no_match_anchors = False
	not_enough_neg_anchors = False

	if total_pos == 0:
		# print('!', end='') # No match anchors
		no_match_anchors = True

	if total_pos > total_fg:
		np.random.shuffle(pos_indices)
		remove_pos_indices = pos_indices[total_fg:]
		anchor_type1d[remove_pos_indices] = len(bboxes)+2

	else:
		total_fg = total_pos
		total_bg = anchor_sampling - total_pos

	total_selected_neg = total_neg
	total_selected_zero_neg = total_zero_neg

	if total_neg > total_bg//2:
		total_selected_zero_neg = min(total_bg//2, total_zero_neg)
		total_selected_neg = min(total_bg - total_selected_zero_neg, total_neg)

	if total_zero_neg > total_bg//2:
		total_selected_neg = min(total_bg//2, total_neg)
		total_selected_zero_neg = min(total_bg - total_selected_neg, total_zero_neg)

	if total_selected_neg + total_selected_zero_neg < total_bg:
		# print('<', end='') # Not enough negative anchors
		not_enough_neg_anchors = True

	# if no_match_anchors is False:
	# 	print(total_fg, total_bg)
		
	np.random.shuffle(neg_indices)
	anchor_type1d[neg_indices[total_selected_neg:]] = len(bboxes)+2

	np.random.shuffle(zero_neg_indices)
	anchor_type1d[zero_neg_indices[:total_selected_zero_neg]] = len(bboxes)

	# Compute loc error
	bboxes.append([0, 0, 0, 0, total_classes])
	bboxes.append([0, 0, 0, 0, total_classes+1])
	bboxes.append([0, 0, 0, 0, total_classes+1])
	matching_2dtensor = tf.gather(params=bboxes, indices=anchor_type1d) # (h*w*k, 5)
	matching_2dtensor = tf.cast(x=matching_2dtensor, dtype='float32')
	matching_bbox_2dtensor = matching_2dtensor[:, :4] # (h*w*k, 4)
	matching_cat_1dtensor = tf.cast(x=matching_2dtensor[:, 4], dtype='int64') # (h*w*k,)
	clz_2dtensor = tf.one_hot(indices=matching_cat_1dtensor, depth=total_classes+1, axis=-1) # (h*w*k, total_classes+1)
	loc_2dtensor = comloc2d(bbox_2dtensor=matching_bbox_2dtensor, abox_2dtensor=abox_2dtensor) # (h*w*k, 4)

	del bboxes[-1]
	del bboxes[-1]
	del bboxes[-1]

	return clz_2dtensor, loc_2dtensor, no_match_anchors, not_enough_neg_anchors

def genxy(dataset, image_dir, ishape, abox_2dtensors, iou_thresholds, total_examples, total_classes, anchor_samplings):
	'''
	Arguments
		anno_file_path:
		image_dir:
		ishape:
		abox_2dtensors:
		iou_thresholds:
		total_classes:
		anchor_samplings
	Return 
		tensor: (h*w*k, 1+4)
	'''

	total_patches 								= total_examples//2
	total_patches_w_augcolor 					= total_examples//4
	total_nests 								= total_examples//8
	total_nests_w_augcolor 						= total_examples - (total_patches+total_patches_w_augcolor+total_nests)
	modes = np.concatenate([np.zeros(total_patches), np.ones(total_nests), 2*np.ones(total_patches_w_augcolor), 3*np.ones(total_nests_w_augcolor)], axis=-1)
	np.random.shuffle(modes)
	
	for i in range(total_examples+100):
		four_images = []
		four_bboxes = []
		
		for _ in range(4):
			image_id, bboxes = dataset[randint(0, len(dataset)-1)]
			image = io.imread(image_dir + '/' + image_id + '.jpg')
			four_images.append(image)
			four_bboxes.append(bboxes)

		image, bboxes = create_image(images=four_images, anno=four_bboxes, ishape=ishape, mode=modes[i])

		for i in range(len(bboxes)):
			bboxes[i] = bboxes[i][:4]+[0]

		tire1_clz_2dtensor, tire1_loc_2dtensor, no_match_anchors1, _ = gentiery(
			bboxes=bboxes, 
			abox_2dtensor=abox_2dtensors[0], 
			iou_thresholds=iou_thresholds[0], 
			total_classes=total_classes, 
			anchor_sampling=anchor_samplings[0])
		tire2_clz_2dtensor, tire2_loc_2dtensor, no_match_anchors2, _ = gentiery(
			bboxes=bboxes, 
			abox_2dtensor=abox_2dtensors[1], 
			iou_thresholds=iou_thresholds[1], 
			total_classes=total_classes, 
			anchor_sampling=anchor_samplings[1])
		tire3_clz_2dtensor, tire3_loc_2dtensor, no_match_anchors3, _ = gentiery(
			bboxes=bboxes, 
			abox_2dtensor=abox_2dtensors[2], 
			iou_thresholds=iou_thresholds[2], 
			total_classes=total_classes, 
			anchor_sampling=anchor_samplings[2])
		tire4_clz_2dtensor, tire4_loc_2dtensor, no_match_anchors4, _ = gentiery(
			bboxes=bboxes, 
			abox_2dtensor=abox_2dtensors[3], 
			iou_thresholds=iou_thresholds[3], 
			total_classes=total_classes, 
			anchor_sampling=anchor_samplings[3])

		if no_match_anchors1 is True and no_match_anchors2 is True and no_match_anchors3 is True and no_match_anchors4 is True:
			print('!', end='')
			continue

		if no_match_anchors1 is True:
			tire1_clz_2dtensor *= 0
		if no_match_anchors2 is True:
			tire2_clz_2dtensor *= 0
		if no_match_anchors3 is True:
			tire3_clz_2dtensor *= 0
		if no_match_anchors4 is True:
			tire4_clz_2dtensor *= 0

		tier1_y = tf.concat(values=[tire1_clz_2dtensor, tire1_loc_2dtensor], axis=-1) # (h1*w1*k1, total_classes+1+4)
		tier2_y = tf.concat(values=[tire2_clz_2dtensor, tire2_loc_2dtensor], axis=-1) # (h2*w2*k2, total_classes+1+4)
		tier3_y = tf.concat(values=[tire3_clz_2dtensor, tire3_loc_2dtensor], axis=-1) # (h3*w3*k3, total_classes+1+4)
		tier4_y = tf.concat(values=[tire4_clz_2dtensor, tire4_loc_2dtensor], axis=-1) # (h4*w4*k4, total_classes+1+4)

		batchy_2dtensor = tf.concat(values=[tier1_y, tier2_y, tier3_y, tier4_y], axis=0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, total_classes+1+4)
		batchx_4dtensor = tf.constant(value=[image], dtype='float32')

		yield batchx_4dtensor, batchy_2dtensor, bboxes

def load_dataset(anno_file_path):
	'''
	'''

	anno_file = open(anno_file_path, 'r')

	lines = anno_file.readlines()
	total_lines = len(lines)
	# print('\nTotal lines: {}'.format(total_lines))

	dataset = []
	for i in range(total_lines):
		image_id, bboxes = parse_line(line=lines[i])
		dataset.append([image_id, bboxes])

	return dataset

def genbbox(dataset, image_dir, ishape, total_examples):
	'''
	'''

	total_patches 					= total_examples//2
	total_patches_w_augcolor 		= total_examples//4
	total_nests 					= total_examples//8
	total_nests_w_augcolor 			= total_examples - (total_patches+total_patches_w_augcolor+total_nests)
	modes = np.concatenate([np.zeros(total_patches), np.ones(total_nests), 2*np.ones(total_patches_w_augcolor), 3*np.ones(total_nests_w_augcolor)], axis=-1)
	np.random.shuffle(modes)
	
	for i in range(total_examples):
		four_images = []
		four_bboxes = []
		
		for _ in range(4):
			image_id, bboxes = dataset[randint(0, len(dataset)-1)]
			image = io.imread(image_dir + '/' + image_id + '.jpg')
			four_images.append(image)
			four_bboxes.append(bboxes)

		image, bboxes = create_image(images=four_images, anno=four_bboxes, ishape=ishape, mode=modes[i])

		yield image, bboxes


