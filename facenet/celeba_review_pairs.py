import matplotlib.pyplot as plt
import json
import numpy as np
from celeba_datagen import genpairs


img_dir_path = '../datasets/CelebA/img_align_celeba/'
anno_file_path = 'anno/celeba_test2_anno.txt'
ishape = [140, 128, 3]
total_identities = 200
total_images = 3
total_examples = 1000
difference_rate = 0.5

batchx4d, batchy1d = genpairs(
	anno_file_path=anno_file_path,
	img_dir_path=img_dir_path,
	ishape=ishape,
	total_identities=total_identities,
	total_images=total_images,
	total_examples=total_examples,
	difference_rate=difference_rate)

batchx5d = batchx4d.reshape((total_examples, 2, ishape[0], ishape[1], ishape[2]))

for exp_idx in range(total_examples):
	x1 = batchx5d[exp_idx, 0]
	x2 = batchx5d[exp_idx, 1]
	y = batchy1d[exp_idx]

	fig, ax = plt.subplots(1, 2, figsize=(15, 7.35))
	fig.suptitle(int(y), fontsize=16)
	ax[0].imshow(x1/255)
	ax[1].imshow(x2/255)
	plt.show()






		