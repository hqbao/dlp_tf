import numpy as np
from random import randint
from skimage import io, transform
from scipy.stats import multivariate_normal


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

def shine(image, ishape):
	'''
	'''

	pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
	center_y = randint(0, ishape[0])
	center_x = randint(0, ishape[1])
	rv = multivariate_normal(mean=[center_y, center_x], cov=randint(0, 10000), allow_singular=True)
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

def flip(image, landmark, size, mode):
	'''
	'''

	if mode == 0:
		image = np.fliplr(image)
		landmark[1] = size[1] - landmark[1]
		landmark[3] = size[1] - landmark[3]
		landmark[5] = size[1] - landmark[5]
		landmark[7] = size[1] - landmark[7]
		landmark[9] = size[1] - landmark[9]

		landmark = [
			landmark[2], landmark[3], landmark[0], landmark[1], 
			landmark[4], landmark[5], 
			landmark[8], landmark[9], landmark[6], landmark[7]
		]

	return image, landmark

def randcrop(image, landmark, size):
	'''
	'''

	max_top_padding = image.shape[0] - size[0]
	max_left_padding = image.shape[1] - size[1]
	origin_y = randint(0, max_top_padding)
	origin_x = randint(0, max_left_padding)

	image = image[origin_y:origin_y+size[0], origin_x:origin_x+size[1], :]

	landmark[0] -= origin_y
	landmark[2] -= origin_y
	landmark[4] -= origin_y
	landmark[6] -= origin_y
	landmark[8] -= origin_y

	landmark[1] -= origin_x
	landmark[3] -= origin_x
	landmark[5] -= origin_x
	landmark[7] -= origin_x
	landmark[9] -= origin_x

	return image, landmark

def genheatmaps(image, landmark, size):
	'''
	'''

	image = np.mean(image, axis=-1, keepdims=True)
	heatmap3d = np.zeros((size[0], size[1], len(landmark)//2), dtype='float32')
	for p in range(5):
		py, px = landmark[2*p:2*p+2]
		pos = np.dstack(np.mgrid[0:size[0]:1, 0:size[1]:1])
		rv = multivariate_normal(mean=[py, px], cov=32)
		heatmap2d = rv.pdf(pos)
		heatmap2d /= np.max(heatmap2d)
		heatmap2d *= 255
		heatmap2d += image[:, :, 0]
		heatmap3d[:, :, p] = heatmap2d

	return heatmap3d

def load_dataset(anno_file_path):
	'''
	'''

	anno_file = open(anno_file_path, 'r')

	lines = anno_file.readlines()
	total_lines = len(lines)
	# print('\nTotal lines: {}'.format(total_lines))

	dataset = []
	for i in range(total_lines):
		line = lines[i]
		anno = line[:-1].split(' ')
		image_file_name, ay, ax, by, bx, cy, cx, dy, dx, ey, ex = anno
		dataset.append([image_file_name, int(ay), int(ax), int(by), int(bx), int(cy), int(cx), int(dy), int(dx), int(ey), int(ex)])

	return dataset

def genxy(dataset, image_dir_path, ishape, total_batches, batch_size):
	'''
	'''

	# np.random.shuffle(dataset)

	for i in range(total_batches):
		batchx4d = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]), dtype='float32')
		batchy4d = np.zeros((batch_size, ishape[0], ishape[1], 5), dtype='float32')

		for j in range(batch_size):
			image_file_name, ay, ax, by, bx, cy, cx, dy, dx, ey, ex = dataset[i*batch_size+j]
			pix = io.imread(image_dir_path+'/'+image_file_name)
			pix = np.array(pix, dtype='float32')

			# Crop
			pix, landmark = randcrop(image=pix, landmark=[ay, ax, by, bx, cy, cx, dy, dx, ey, ex], size=ishape[:2])

			# Color augmentation
			pix = augcolor(image=pix, ishape=ishape)

			# Flip
			pix, landmark = flip(image=pix, landmark=landmark, size=ishape[:2], mode=randint(0, 1))

			# Black & white
			pix = np.mean(pix, axis=-1, keepdims=True)

			heatmap3d = genheatmaps(image=pix, landmark=landmark, size=ishape[:2])

			batchx4d[j, :, :, :] = pix
			batchy4d[j] = heatmap3d

		yield batchx4d, batchy4d

