import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datagen import genheatmap


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mode = 'train'
anno_file_path = '../datasets/wider_face/full_'+mode+'_anno.txt'
img_dir_path = '../datasets/wider_face/full_'+mode+'_images'
ishape = [448, 448, 1]
total_examples = 12678 # 12678, 3164
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

		_, ax = plt.subplots(1, 2, figsize=(15, 7.35))
		ax[0].imshow(x[:, :, 0]/255, vmin=0, vmax=1)
		ax[1].imshow(heatmap3d[:, :, 0], vmin=0, vmax=1)
		plt.show()






		