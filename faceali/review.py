import numpy as np
import matplotlib.pyplot as plt
from datagen import load_dataset, genxy


anno_file_path = '/Users/baulhoa/Documents/PythonProjects/datasets/faceali/train.txt'
image_dir_path = '/Users/baulhoa/Documents/PythonProjects/datasets/faceali/train'
ishape = [112, 112, 1]
total_examples = 202599 # 202599
batch_size = 100
total_classes = 3000
total_batches = total_examples//batch_size

dataset = load_dataset(anno_file_path=anno_file_path)

gen = genxy(
	dataset=dataset, 
	image_dir_path=image_dir_path, 
	ishape=ishape, 
	total_batches=total_batches, 
	batch_size=batch_size,
	total_classes=total_classes)

for _ in range(total_batches):
	batchx4d, batchy4d = next(gen)

	for i in range(int(batchx4d.shape[0])):
		pix = batchx4d[i, :, :, 0]
		heatmap3d = batchy4d[i]

		_, ax = plt.subplots(1, 6, figsize=(15, 7.35))
		ax[0].imshow(np.array(pix, dtype='uint8'))
		
		ax[1].imshow(heatmap3d[:, :, 0])
		ax[2].imshow(heatmap3d[:, :, 1])
		ax[3].imshow(heatmap3d[:, :, 2])
		ax[4].imshow(heatmap3d[:, :, 3])
		ax[5].imshow(heatmap3d[:, :, 4])

		plt.show()
