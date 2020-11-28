import numpy as np
import tensorflow as tf
from datetime import datetime
from vggface2_datagen import genxyclz
from model import build_clz_model


img_dir_path = '../datasets/vggface2/train_refined_resized'
anno_file_path = 'anno/vggface2_train1_anno.txt'
test_anno_file_path = 'anno/vggface2_test1_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [1, 1]]]
dense_settings = []
total_epoches = 1000
total_identities = 3000
total_examples = 840000
total_test_examples = 60000
batch_size = 400
total_batches = total_examples//batch_size
total_test_batches = total_test_examples//batch_size

log_file_name = "{}:train-clz-log.txt".format(datetime.now().time())
min_loss = 2**32

model = build_clz_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings, 
	total_classes=total_identities)
# model.summary()
# model.load_weights('{}/clz/weights.h5'.format(output_path), by_name=True)

for epoch in range(total_epoches):
	# Train
	gen = genxyclz(
		anno_file_path=anno_file_path, 
		img_dir_path=img_dir_path, 
		ishape=ishape, 
		total_batches=total_batches, 
		batch_size=batch_size,
		total_classes=total_identities)

	loss = []

	print('\nTrain epoch: {}'.format(epoch), end='')

	for batch_idx in range(total_batches):
		batchx4d, batchy2d = next(gen)
		batch_loss = model.train_on_batch(batchx4d, batchy2d)
		loss.append(batch_loss)

		if batch_idx%100 == 0:
			print('-', end='')

	mean_loss = np.mean(loss, axis=0)
	print(round(mean_loss, 2), end='')

	model.save_weights('{}/clz/_weights.h5'.format(output_path))

	with open('{}/clz/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}, '.format(round(mean_loss, 2)))

	# Validate
	gen = genxyclz(
		anno_file_path=test_anno_file_path, 
		img_dir_path=img_dir_path, 
		ishape=ishape, 
		total_batches=total_test_batches, 
		batch_size=batch_size,
		total_classes=total_identities)

	loss = []

	print(', validate', end='')

	for batch_idx in range(total_test_batches):
		batchx4d, batchy2d = next(gen)
		batch_loss = model.test_on_batch(batchx4d, batchy2d)
		loss.append(batch_loss)

		if batch_idx%100 == 0:
			print('-', end='')

	mean_loss = np.mean(loss, axis=0)
	if mean_loss < min_loss:
		min_loss = mean_loss
		model.save_weights('{}/clz/weights.h5'.format(output_path))

	print('{}/{}'.format(round(mean_loss, 2), round(min_loss, 5)), end='')

	with open('{}/clz/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}\n'.format(round(mean_loss, 2)))










