import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from celeba_datagen import genlm
from model import build_ali_model


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

img_dir_path = '../datasets/CelebA/modified_celeba'
test_anno_file_path = 'anno/celeba_test_landmark_anno.txt'
output_path = 'output'
ishape = [112, 112, 1]
total_test_examples = 3000 # 3000
batch_size = 1
total_test_batches = total_test_examples//batch_size

model = build_ali_model(ishape=ishape, mode='train')
# model.summary()
model.load_weights('{}/ali/weights.h5'.format(output_path), by_name=True)

gen = genlm(
	anno_file_path=test_anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_batches=total_test_examples, 
	batch_size=batch_size)

for _ in range(total_test_batches):
	batchx4d, landmark2d, _ = next(gen)
	print('Start: {}'.format(datetime.now().time()))
	prediction = model.predict_on_batch(batchx4d) # (batch_size, h, w, 5)
	print('End: {}'.format(datetime.now().time()))

	heatmap4d = prediction.numpy()
	heatmap4d = heatmap4d[:, :, :, :] - batchx4d[:, :, :, 0:1]/255
	heatmap4d = np.where(heatmap4d > 0.0, heatmap4d, 0)

	hm12d = heatmap4d[:, :, :, 0]
	hm22d = heatmap4d[:, :, :, 1]
	hm32d = heatmap4d[:, :, :, 2]
	hm42d = heatmap4d[:, :, :, 3]
	hm52d = heatmap4d[:, :, :, 4]

	for i in range(batch_size):
		_, ax = plt.subplots(1, 6, figsize=(15, 7.35))
		ax[0].imshow(batchx4d[i, :, :, 0]/255)

		landmark = landmark2d[i]
		for p in range(5):
			y, x = landmark[2*p:2*p+2]
			ax[0].scatter(x, y, s=5, c='cyan')

		ax[1].imshow(hm12d[i])
		ax[2].imshow(hm22d[i])
		ax[3].imshow(hm32d[i])
		ax[4].imshow(hm42d[i])
		ax[5].imshow(hm52d[i])

		hm1 = hm12d[i]
		a = np.argmax(hm1)
		ya = a//ishape[0]
		xa = a%ishape[0]
		ax[0].scatter(xa, ya, s=5, c='red')

		hm2 = hm22d[i]
		b = np.argmax(hm2)
		ya = b//ishape[0]
		xa = b%ishape[0]
		ax[0].scatter(xa, ya, s=5, c='red')

		hm3 = hm32d[i]
		c = np.argmax(hm3)
		yc = c//ishape[0]
		xc = c%ishape[0]
		ax[0].scatter(xc, yc, s=5, c='red')

		hm4 = hm42d[i]
		d = np.argmax(hm4)
		yd = d//ishape[0]
		xd = d%ishape[0]
		ax[0].scatter(xd, yd, s=5, c='red')

		hm5 = hm52d[i]
		e = np.argmax(hm5)
		ye = e//ishape[0]
		xe = e%ishape[0]
		ax[0].scatter(xe, ye, s=5, c='red')

		plt.show()






