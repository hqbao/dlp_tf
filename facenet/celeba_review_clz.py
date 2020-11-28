import matplotlib.pyplot as plt
from celeba_datagen import genxyclz


anno_file_path = 'anno/celeba_train1_anno.txt'
img_dir_path = '../datasets/CelebA/img_align_celeba/'
ishape = [140, 128, 3]
total_examples = 25000 # 3000
batch_size = 1000
total_batches = total_examples//batch_size

gen = genxyclz(
	anno_file_path=anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_examples=total_examples, 
	batch_size=batch_size)

for _ in range(total_batches):
	batchx4d, id_batchy1d = next(gen)

	for i in range(int(batchx4d.shape[0])):
		x = batchx4d[i]
		id = int(id_batchy1d[i])

		# if id != 1:
		# 	continue

		_, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(x/256)
		ax.set_xlabel('Oder: {}, ID: {}'.format(i, id))
		plt.show()






		