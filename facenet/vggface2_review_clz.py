import numpy as np
import matplotlib.pyplot as plt
from vggface2_datagen import genxyclz


anno_file_path = 'anno/vggface2_test1_anno.txt'
img_dir_path = '../datasets/vggface2/train_refined_resized'
ishape = [112, 112, 3]
total_examples = 336000 # 24000
batch_size = 100
total_identities = 3000
total_batches = total_examples//batch_size

gen = genxyclz(
	anno_file_path=anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_batches=total_batches, 
	batch_size=batch_size,
	total_classes=total_identities)

for _ in range(total_batches):
	batchx4d, batchy2d = next(gen)

	for i in range(int(batchx4d.shape[0])):
		x = batchx4d[i]
		y = batchy2d[i]
		id = np.argmax(y)

		# if id != 2:
		# 	continue

		_, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(x/255)
		ax.set_xlabel('Oder: {}, ID: {}'.format(i, id))
		plt.show()