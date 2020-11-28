import skimage.io as io
import numpy as np
from random import randint


def genxyclz(anno_file_path, img_dir_path, ishape, total_batches, batch_size, total_classes):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	total_lines = len(lines)
	print('\nTotal exmples (lines): {}'.format(total_lines))
	# np.random.shuffle(lines)

	for i in range(total_batches):
		batchx4d = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]), dtype='float32')
		batchy2d = np.zeros((batch_size, total_classes), dtype='float32')

		for j in range(batch_size):
			line_idx = i*batch_size + j
			line = lines[line_idx][:-1]
			anno = line.split(' ')
			id_folder, image_file_name = anno[0].split('/')
			id = int(anno[1])
			image_file_path = img_dir_path + '/' + id_folder + '/' + image_file_name
			x = io.imread(image_file_path)

			# flip
			if randint(0, 1) == 0:
				x = np.fliplr(x)

			# crop
			h = x.shape[0] - ishape[0]
			w = x.shape[1] - ishape[1]
			origin_y = randint(0, h)
			origin_x = randint(0, w)
			x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]

			batchx4d[j, :, :, :] = x
			batchy2d[j, id] = 1

		yield batchx4d, batchy2d

def gentriplets(anno_file_path, img_dir_path, ishape, total_identities, total_images, total_examples, batch_size):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	total_lines = len(lines)
	print('\nTotal lines: {}'.format(total_lines))

	yx_dist = {}

	for i in range(total_lines):
		line = lines[i][:-1]
		anno = line.split(' ')
		image_file = anno[0]
		id = int(anno[1])

		if id not in yx_dist:
			yx_dist[id] = [image_file]
		else:
			yx_dist[id].append(image_file)

	image_set_list = []

	for id in sorted(yx_dist):
		image_set_list.append(yx_dist[id])

	image_set_list = image_set_list[:total_identities]
	np.random.shuffle(image_set_list)
	for i in range(len(image_set_list)):
		np.random.shuffle(image_set_list[i][:total_images])

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
	print('Total triplets: {}'.format(len(triplet_indices)))
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
			anchor = image_set_list[i1][j1]
			positive = image_set_list[i2][j2]
			negative = image_set_list[i3][j3]

			x = io.imread('{}/{}'.format(img_dir_path, anchor))
			h = x.shape[0] - ishape[0]
			w = x.shape[1] - ishape[1]
			origin_y = randint(0, h)
			origin_x = randint(0, w)
			x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
			x_anchor = x

			x = io.imread('{}/{}'.format(img_dir_path, positive))
			h = x.shape[0] - ishape[0]
			w = x.shape[1] - ishape[1]
			origin_y = randint(0, h)
			origin_x = randint(0, w)
			x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
			x_positive = x

			x = io.imread('{}/{}'.format(img_dir_path, negative))
			h = x.shape[0] - ishape[0]
			w = x.shape[1] - ishape[1]
			origin_y = randint(0, h)
			origin_x = randint(0, w)
			x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
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
	print('Total lines: {}'.format(total_lines))

	yx_dist = {}

	for i in range(total_lines):
		line = lines[i][:-1]
		anno = line.split(' ')
		image_file = anno[0]
		id = int(anno[1])

		if id not in yx_dist:
			yx_dist[id] = [image_file]
		else:
			yx_dist[id].append(image_file)

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
			diff_pairs_indices.append([[i, randint(0, total_images-1)], [j, randint(0, total_images-1)]])

	np.random.shuffle(same_pairs_indices)
	np.random.shuffle(diff_pairs_indices)

	total_same_pairs = len(same_pairs_indices)
	total_diff_pairs = len(diff_pairs_indices)

	total_selected_same_pairs = round((1-difference_rate)*total_examples)
	total_selected_diff_pairs = round(difference_rate*total_examples)

	same_pairs_indices = same_pairs_indices[:total_selected_same_pairs]
	diff_pairs_indices = diff_pairs_indices[:total_selected_diff_pairs]
	pairs_indices = same_pairs_indices + diff_pairs_indices

	print('Total pairs: {}/{}'.format(len(pairs_indices), total_same_pairs+total_diff_pairs))

	if len(pairs_indices) < total_examples:
		return
	
	print('Total same pairs: {}'.format(total_selected_same_pairs))
	print('Total diff pairs: {}'.format(total_selected_diff_pairs))

	batchx4d = np.zeros((2*total_examples, ishape[0], ishape[1], ishape[2]))
	batchy1d = np.zeros(total_examples)

	for i in range(total_examples):
		[anchor_y, anchor_x], [versus_y, versus_x] = pairs_indices[i]
		anchor = image_set_list[anchor_y][anchor_x]
		versus = image_set_list[versus_y][versus_x]

		x = io.imread('{}/{}'.format(img_dir_path, anchor))
		h = x.shape[0] - ishape[0]
		w = x.shape[1] - ishape[1]
		origin_y = randint(0, h)
		origin_x = randint(0, w)
		x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
		x_anchor = x

		x = io.imread('{}/{}'.format(img_dir_path, versus))
		h = x.shape[0] - ishape[0]
		w = x.shape[1] - ishape[1]
		origin_y = randint(0, h)
		origin_x = randint(0, w)
		x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
		x_versus = x

		batchx4d[2*i] = x_anchor
		batchx4d[2*i+1] = x_versus
		batchy1d[i] = anchor_y == versus_y

	return batchx4d, batchy1d

def genid(anno_file_path, img_dir_path, ishape, total_identities, total_same_identities):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	total_images = len(lines)

	print('Total images: {}'.format(total_images))

	yx_dist = {}

	for i in range(total_images):
		line = lines[i][:-1]
		anno = line.split(' ')
		id_folder, file_name = anno[0].split('/')
		id = int(anno[1])

		if id not in yx_dist:
			yx_dist[id] = [id_folder + '/' + file_name]
		else:
			yx_dist[id].append(id_folder + '/' + file_name)

	yx = []
	for identity in sorted(yx_dist):
		yx.append(yx_dist[identity])

	print('Total identities: {}'.format(len(yx)))
	yx = yx[:total_identities]
	indices = np.arange(total_identities)
	np.random.shuffle(indices)

	batchx4d = np.zeros((total_identities*total_same_identities, ishape[0], ishape[1], ishape[2]))
	batchy1d = np.zeros(total_identities*total_same_identities)

	for i in indices:
		id = indices[i]
		for j in range(total_same_identities):
			image_file = yx[id][j]
			x = io.imread('{}/{}'.format(img_dir_path, image_file))
			h = x.shape[0] - ishape[0]
			w = x.shape[1] - ishape[1]
			origin_y = randint(0, h)
			origin_x = randint(0, w)
			x = x[origin_y:origin_y+ishape[0], origin_x:origin_x+ishape[1], :]
			batchx4d[i*total_same_identities+j] = x

		batchy1d[i*total_same_identities:i*total_same_identities+total_same_identities] = id

	return batchx4d, batchy1d






