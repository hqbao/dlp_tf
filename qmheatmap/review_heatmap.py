import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datagen import genheatmap


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mode = 'train'
anno_file_path = '../datasets/quizmarker/'+mode+'_anno.txt'
img_dir_path = '../datasets/quizmarker/full_images'
ishape = [264, 224, 1]
total_examples = 800 # 978
batch_size = 1
total_batches = total_examples//batch_size

gen = genheatmap(
	anno_file_path=anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_batches=total_batches, 
	batch_size=batch_size)

for _ in range(total_batches):
	print('Start: {}'.format(datetime.now().time()))
	batchx4d, heatmap4d = next(gen)
	print('End: {}'.format(datetime.now().time()))
	heatmap4d -= batchx4d/255
	# heatmap4d = np.where(heatmap4d > 0.8, heatmap4d, 0)

	for i in range(int(batchx4d.shape[0])):
		x = batchx4d[i]
		heatmap3d = heatmap4d[i]

		_, ax = plt.subplots(1, 4, figsize=(15, 7.35))
		ax[0].imshow(x[:, :, 0]/255)
		ax[1].imshow(heatmap3d[:, :, 0], vmin=0, vmax=1)
		ax[2].imshow(heatmap3d[:, :, 1], vmin=0, vmax=1)
		mergehm = np.sum(heatmap3d, axis=-1)
		ax[3].imshow(mergehm, vmin=0, vmax=1)
		plt.show()






		