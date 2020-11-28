import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datagen import genheatmap
from model import build_model


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mode = 'test'
test_anno_file_path = '../datasets/quizmarker/'+mode+'_anno.txt'
test_img_dir_path = '../datasets/quizmarker/full_images'
output_path = 'output'
ishape = [272, 224, 1]
total_test_examples = 100
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
	prediction = model.predict_on_batch(batchx4d) # (batch_size, h, w, 2)
	print('End: {}'.format(datetime.now().time()))

	heatmap4d = prediction.numpy()
	heatmap4d -= batchx4d/255

	for i in range(batch_size):
		x = batchx4d[i]
		hm3d = heatmap4d[i]

		hm1 = hm3d[:, :, 0]
		# hm1 = hm1/np.max(hm1)
		# hm1 = np.where(hm1 > 0.8, hm1, 0)

		hm2 = hm3d[:, :, 1]
		# hm2 = hm2/np.max(hm2)
		# hm2 = np.where(hm2 > 0.8, hm2, 0)

		_, ax = plt.subplots(1, 3, figsize=(15, 7.35))
		ax[0].imshow(x[:, :, 0]/255, vmin=0, vmax=1)
		ax[1].imshow(hm1, vmin=0, vmax=1)
		ax[2].imshow(hm2, vmin=0, vmax=1)
		plt.show()






