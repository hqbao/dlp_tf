import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datagen import genheatmap
from model import build_model


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

test_anno_file_path = '../datasets/wider_face/full_test_anno.txt'
test_img_dir_path = '../datasets/wider_face/full_test_images'
output_path = 'output'
ishape = [448, 448, 1]
total_test_examples = 3164 # 3164
batch_size = 1
total_test_batches = total_test_examples//batch_size

model = build_model(ishape=ishape, mode='test')
# model.summary()
model.load_weights('{}/weights.h5'.format(output_path), by_name=True)

gen = genheatmap(
	anno_file_path=test_anno_file_path, 
	img_dir_path=test_img_dir_path, 
	ishape=ishape, 
	total_batches=total_test_examples, 
	batch_size=batch_size)

for _ in range(total_test_batches):
	batchx4d, _ = next(gen)
	print('Start: {}'.format(datetime.now().time()))
	prediction = model.predict_on_batch(batchx4d) # (batch_size, h, w, 5)
	print('End: {}'.format(datetime.now().time()))

	heatmap4d = prediction.numpy()
	heatmap4d -= batchx4d/255
	heatmap3d = heatmap4d[:, :, :, 0]

	for i in range(batch_size):
		hm12d = heatmap3d[i]
		# hm12d = hm12d/np.max(hm12d)
		# hm12d = np.where(hm12d > 0.8, hm12d, 0)

		_, ax = plt.subplots(1, 2, figsize=(15, 7.35))
		ax[0].imshow(batchx4d[i, :, :, 0]/255, vmin=0, vmax=1)
		ax[1].imshow(hm12d, vmin=0, vmax=1)
		plt.show()






