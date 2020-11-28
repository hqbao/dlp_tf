import numpy as np
import matplotlib.pyplot as plt
from celeba_datagen import genlm


anno_file_path = 'anno/celeba_train_landmark_anno.txt'
img_dir_path = '../datasets/CelebA/modified_celeba'
ishape = [112, 112, 1]
total_examples = 25000 # 25000, 3000
batch_size = 100
total_batches = total_examples//batch_size

gen = genlm(
	anno_file_path=anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_batches=total_batches, 
	batch_size=batch_size)

for _ in range(total_batches):
	batchx4d, landmark2d, heatmap4d = next(gen)
	# heatmap4d -= batchx4d[:, :, :, :]/255

	for i in range(int(batchx4d.shape[0])):
		landmark = landmark2d[i]

		_, ax = plt.subplots(1, 4, figsize=(15, 7.35))
		ax[0].imshow(batchx4d[i, :, :, 0]/255)

		landmark = landmark2d[i]
		for p in range(5):
			y, x = landmark[2*p:2*p+2]
			ax[0].scatter(x, y, s=5, c='cyan')

		ax[1].imshow(heatmap4d[i, :, :, 0])
		ax[2].imshow(heatmap4d[i, :, :, 1])
		ax[3].imshow(heatmap4d[i, :, :, 2])
		plt.show()






		