import skimage.io as io
import numpy as np
from random import randint, shuffle
from scipy.stats import multivariate_normal

def genheatmap(anno_file_path, img_dir_path, ishape, total_batches, batch_size):
	'''
	'''

	anno_file = open(anno_file_path, 'r')
	lines = anno_file.readlines()
	np.random.shuffle(lines)
	total_lines = len(lines)
	# print('\nTotal lines: {}'.format(total_lines))

	for batch_idx in range(total_batches):
		batchx4d = np.zeros((batch_size, ishape[0], ishape[1], ishape[2]), dtype='int32')
		heatmap4d = np.zeros((batch_size, ishape[0], ishape[1], 1), dtype='float32')

		for i in range(batch_size):
			line_idx = batch_idx*batch_size+i
			line = lines[line_idx]
			anno = line[:-1].split(' ')
			image_id = anno[0]
			anno = anno[1:]
			bboxatt_list = [list(map(float, anno[i:i+10])) for i in range(0, len(anno), 10)]

			# Take image data
			x = io.imread('{}/{}.jpg'.format(img_dir_path, image_id))
			x = np.sum(x, axis=-1, keepdims=True)
			x = x/3

			# Crop to ishape
			h, w, _ = x.shape
			crop_y1 = randint(0, h - ishape[0])
			crop_x1 = randint(0, w - ishape[1])
			crop_y2 = crop_y1 + ishape[0]
			crop_x2 = crop_x1 + ishape[1]
			x = x[crop_y1:crop_y2, crop_x1:crop_x2, :]

			batchx4d[i] = x

			# Filter
			bboxes = []
			for y1, x1, y2, x2, blur, expression, illumination, invalid, occlusion, pose in bboxatt_list:
				if expression not in [0, 1]:
					continue

				if illumination not in [0, 1]:
					continue

				if invalid not in [0, 1]:
					continue

				if occlusion not in [0, 1, 2]:
					continue

				if occlusion not in [0, 1, 2]:
					continue

				if pose not in [0, 1]:
					continue

				if y2 - crop_y1 < 0.5*(y2 - y1):
					continue

				if x2 - crop_x1 < 0.5*(x2 - x1):
					continue

				if crop_y2 - y1 < 0.5*(y2 - y1):
					continue

				if crop_x2 - x1 < 0.5*(x2 - x1):
					continue

				bboxes.append([int(y1-crop_y1), int(x1-crop_x1), int(y2-crop_y1), int(x2-crop_x1)])

			if len(bboxes) == 0:
				continue

			# Generate heatmap
			heatmap3d = np.zeros((ishape[0], ishape[1], 1), dtype='float32')
			bbox2d = np.array(bboxes)
			for b in range(bbox2d.shape[0]):
				y1, x1, y2, x2 = bbox2d[b]
				center_y = y1 + 0.5*(y2 - y1)
				center_x = x1 + 0.5*(x2 - x1)
				a = (y2 - y1)*(x2 - x1)

				pos = np.dstack(np.mgrid[0:ishape[0]:1, 0:ishape[1]:1])
				rv = multivariate_normal(mean=[center_y, center_x], cov=0.1*a, allow_singular=True)
				heatmap2d = rv.pdf(pos)
				heatmap2d /= np.max(heatmap2d)
				heatmap3d[:, :, 0] += heatmap2d

			heatmap3d += x/255 # Heatmaps are added to image
			heatmap4d[i] = heatmap3d

		yield batchx4d, heatmap4d



