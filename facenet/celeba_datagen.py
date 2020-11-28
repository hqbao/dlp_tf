import skimage.io as io
import tensorflow as tf
import numpy as np
import cv2
from random import randint, shuffle
from scipy.stats import multivariate_normal


def genxyclz(anno_file_path, img_dir_path, ishape, total_examples, batch_size):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	np.random.shuffle(lines)
	total_lines = len(lines)
	# print('Total lines: {}'.format(total_lines))
	if total_examples > total_lines:
		return

	images = []
	id1d = np.zeros(total_lines)
	bbox2d = np.zeros((total_lines, 4), dtype='int64')

	for i in range(total_lines):
		line = lines[i][:-1]
		anno = line.split(' ')

		images.append(anno[0])

		identity = int(anno[1])
		id1d[i] = identity

		bbox = list(map(int, anno[2:]))
		bbox2d[i] = bbox

	# print('Total identities: {}'.format(np.unique(id1d).shape[0]))

	total_batches = total_examples//batch_size

	for batch_idx in range(total_batches):
		batchx4d = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]))
		id_batchy1d = np.zeros(batch_size, dtype='float32')

		for i in range(batch_size):
			img_file = images[batch_idx*batch_size+i]
			x = io.imread('{}/{}'.format(img_dir_path, img_file))
			y1, x1, y2, x2 = bbox2d[batch_idx*batch_size+i]
			x = x[y1:y2, x1:x2, :]
			x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
			x = np.clip(x, 0, 255)
			
			batchx4d[i, :, :, :] = x
			id_batchy1d[i] = id1d[batch_idx*batch_size+i]

		yield batchx4d, id_batchy1d

def gentriplets(anno_file_path, img_dir_path, ishape, total_identities, total_images, total_examples, batch_size):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	total_lines = len(lines)
	# print('Total lines: {}'.format(total_lines))

	yx_dist = {}

	for i in range(total_lines):
		line = lines[i][:-1]
		anno = line.split(' ')
		image_file = anno[0]
		id = int(anno[1])
		bbox = list(map(int, anno[2:]))

		if id not in yx_dist:
			yx_dist[id] = [[image_file] + bbox]
		else:
			yx_dist[id].append([image_file] + bbox)

	image_set_list = []

	for id in sorted(yx_dist):
		image_set_list.append(yx_dist[id])

	image_set_list = image_set_list[:total_identities]

	triplet_indices = []
	for i in range(total_identities-1):
		for j in range(total_images-1):
			for k in range(j+1, total_images):
				for l in range(i+1, total_identities):
					triplet_indices.append([
						[i, j], # anchor
						[i, k], # positive
						[l, randint(0, total_images-1)] # negative
					])

	np.random.shuffle(triplet_indices)
	# print('Total triplets: {}'.format(len(triplet_indices)))
	if len(triplet_indices) < total_examples:
		return

	triplet_indices = triplet_indices[:total_examples]
	total_triplets = len(triplet_indices)
	total_batches = total_triplets//batch_size
	
	for batch_idx in range(total_batches):
		batchx4d = np.zeros((3*batch_size, ishape[0], ishape[1], ishape[2]))
		batchy1d = np.zeros(3*batch_size, dtype='int64')

		for idx in range(batch_size):
			[i1, j1], [i2, j2], [i3, j3] = triplet_indices[batch_idx*batch_size+idx]
			anchor = image_set_list[i1][j1][0]
			positive = image_set_list[i2][j2][0]
			negative = image_set_list[i3][j3][0]

			x = io.imread('{}/{}'.format(img_dir_path, anchor))
			y1, x1, y2, x2 = image_set_list[i1][j1][1:]
			x = x[y1:y2, x1:x2, :]
			x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
			x = np.clip(x, 0, 255)
			x_anchor = x

			x = io.imread('{}/{}'.format(img_dir_path, positive))
			y1, x1, y2, x2 = image_set_list[i2][j2][1:]
			x = x[y1:y2, x1:x2, :]
			x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
			x = np.clip(x, 0, 255)
			x_positive = x

			x = io.imread('{}/{}'.format(img_dir_path, negative))
			y1, x1, y2, x2 = image_set_list[i3][j3][1:]
			x = x[y1:y2, x1:x2, :]
			x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
			x = np.clip(x, 0, 255)
			x_negative = x

			batchx4d[3*idx+0, :, :, :] = x_anchor
			batchx4d[3*idx+1, :, :, :] = x_positive
			batchx4d[3*idx+2, :, :, :] = x_negative

			batchy1d[3*idx:3*idx+3] = [i1, i2, i3]
		
		yield batchx4d, batchy1d

def genpairs(anno_file_path, img_dir_path, ishape, total_identities, total_images, total_examples, difference_rate=0.8):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	total_lines = len(lines)
	# print('Total lines: {}'.format(total_lines))

	yx_dist = {}

	for i in range(total_lines):
		line = lines[i][:-1]
		anno = line.split(' ')
		image_file = anno[0]
		id = int(anno[1])
		bbox = list(map(int, anno[2:]))

		if id not in yx_dist:
			yx_dist[id] = [[image_file] + bbox]
		else:
			yx_dist[id].append([image_file] + bbox)

	image_set_list = []

	for id in sorted(yx_dist):
		image_set_list.append(yx_dist[id])
	
	image_set_list = image_set_list[:total_identities]

	same_pairs_indices = []
	for i in range(total_identities):
		for j in range(total_images-1):
			for k in range(j+1, total_images):
				same_pairs_indices.append([[i, j], [i, k]])

	diff_pairs_indices = []
	for i in range(total_identities-1):
		for j in range(i+1, total_identities):
			diff_pairs_indices.append([[i, randint(0, 2)], [j, randint(0, 2)]])

	np.random.shuffle(same_pairs_indices)
	np.random.shuffle(diff_pairs_indices)

	total_same_pairs = len(same_pairs_indices)
	total_diff_pairs = len(diff_pairs_indices)

	total_selected_same_pairs = round((1-difference_rate)*total_examples)
	total_selected_diff_pairs = round(difference_rate*total_examples)

	same_pairs_indices = same_pairs_indices[:total_selected_same_pairs]
	diff_pairs_indices = diff_pairs_indices[:total_selected_diff_pairs]
	pairs_indices = same_pairs_indices + diff_pairs_indices

	if len(pairs_indices) < total_examples:
		return

	# print('Total pairs: {}/{}'.format(len(pairs_indices), total_same_pairs+total_diff_pairs))
	# print('Total same pairs: {}'.format(total_selected_same_pairs))
	# print('Total diff pairs: {}'.format(total_selected_diff_pairs))

	batchx4d = np.zeros((2*total_examples, ishape[0], ishape[1], ishape[2]))
	batchy1d = np.zeros(total_examples)

	for i in range(total_examples):
		[anchor_y, anchor_x], [versus_y, versus_x] = pairs_indices[i]
		anchor = image_set_list[anchor_y][anchor_x][0]
		versus = image_set_list[versus_y][versus_x][0]

		x = io.imread('{}/{}'.format(img_dir_path, anchor))
		y1, x1, y2, x2 = image_set_list[anchor_y][anchor_x][1:]
		x = x[y1:y2, x1:x2, :]
		x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
		x = np.clip(x, 0, 255)
		x_anchor = x

		x = io.imread('{}/{}'.format(img_dir_path, versus))
		y1, x1, y2, x2 = image_set_list[versus_y][versus_x][1:]
		x = x[y1:y2, x1:x2, :]
		x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
		x = np.clip(x, 0, 255)
		x_versus = x

		batchx4d[2*i] = x_anchor
		batchx4d[2*i+1] = x_versus
		batchy1d[i] = anchor_y == versus_y

	return batchx4d, batchy1d

def genlm(anno_file_path, img_dir_path, ishape, total_batches, batch_size):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	np.random.shuffle(lines)
	total_lines = len(lines)
	# print('\nTotal lines: {}'.format(total_lines))

	for batch_idx in range(total_batches):
		batchx4d = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]), dtype='float32')
		landmark2d = np.zeros((batch_size, 10), dtype='float32')
		heatmap4d = np.zeros((batch_size, ishape[0], ishape[1], 5), dtype='float32')

		for i in range(batch_size):
			line_idx = batch_idx*batch_size+i
			line = lines[line_idx][:-1]
			anno = line.split(' ')
			image_id = int(anno[0][:-4])
			# identity = int(anno[1])
			bbox = list(map(int, anno[2:6]))
			landmark = np.array(list(map(int, anno[6:])), dtype='float32')

			img = io.imread('{}/{}.jpg'.format(img_dir_path, str(image_id).zfill(6)))
			y1, x1, y2, x2 = bbox
			x = np.sum(img[y1:y2, x1:x2, :], axis=-1, keepdims=True)
			x = x/3.0

			# crop
			h = x.shape[0] - ishape[0]
			w = x.shape[1] - ishape[1]
			origin_y = randint(0, h)
			origin_x = randint(0, w)
			# origin_y = 10
			# origin_x = 30
			x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
			landmark[::2] -= origin_y
			landmark[1::2] -= origin_x

			# flip, should not flip, avoid unexpected symmetric detection
			# if randint(0, 1) == 0:
			# 	x = np.fliplr(x)
			# 	landmark[1::2] = ishape[1] - landmark[1::2]

			batchx4d[i, :, :, :] = x
			landmark2d[i, :] = landmark

			heatmap3d = np.zeros((ishape[0], ishape[1], 5), dtype='float32')
			for p in range(5):
				py, px = landmark[2*p:2*p+2]
				pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
				rv = multivariate_normal(mean=[py, px], cov=32)
				heatmap2d = rv.pdf(pos)
				heatmap2d /= np.max(heatmap2d)
				heatmap3d[:, :, p] = heatmap2d

			heatmap3d /= np.max(heatmap3d)
			heatmap3d += x[:, :, :]/255
			heatmap4d[i] = heatmap3d

		yield batchx4d, landmark2d, heatmap4d



