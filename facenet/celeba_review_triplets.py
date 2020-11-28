import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
from celeba_datagen import gentriplets


img_dir_path = '../datasets/CelebA/img_align_celeba/'
anno_file_path = 'anno/celeba_train2_anno.txt'
ishape = [140, 128, 3]
total_identities = 200
total_images = 25 # images per identity
total_examples = 5970000 # 59700
batch_size = 500
total_batches = total_examples//batch_size

gen = gentriplets(
	anno_file_path=anno_file_path,
	img_dir_path=img_dir_path,
	ishape=ishape,
	total_identities=total_identities,
	total_images=total_images,
	total_examples=total_examples,
	batch_size=batch_size)

for _ in range(total_batches):
	batchx4d, _ = next(gen)
	print(batchx4d.shape)
	batchx5d = batchx4d.reshape((batch_size, 3, ishape[0], ishape[1], ishape[2]))

	for exp_idx in range(int(batchx5d.shape[0]/8)):
		plt.figure(figsize=(15, 7.35))
		for i in range(8):
			plt.subplot(4, 6, 3*i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(batchx5d[exp_idx*8+i][0]/255)

			plt.subplot(4, 6, 3*i+2)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(batchx5d[exp_idx*8+i][1]/255)

			plt.subplot(4, 6, 3*i+3)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(batchx5d[exp_idx*8+i][2]/255)
			
		plt.show()





		