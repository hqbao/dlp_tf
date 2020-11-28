import numpy as np
import tensorflow as tf
from datetime import datetime
from datagen import genheatmap
from model import build_model


anno_file_path = '../datasets/quizmarker/train_anno.txt'
img_dir_path = '../datasets/quizmarker/full_images'
test_anno_file_path = '../datasets/quizmarker/train_anno.txt'
test_img_dir_path = '../datasets/quizmarker/full_images'
output_path = 'output'
ishape = [272, 224, 1]
total_epoches = 1000
total_examples = 1 # 800
total_test_examples = 1
batch_size = 1 # 100
total_batches = total_examples//batch_size
total_test_batches = total_test_examples//batch_size

log_file_name = "{}:train-face-log.txt".format(datetime.now().time())
min_loss = 2**32

model = build_model(ishape=ishape)
# model.summary()
model.load_weights('{}/_weights.h5'.format(output_path), by_name=True)

for epoch in range(total_epoches):
	# Train
	gen = genheatmap(
		anno_file_path=anno_file_path, 
		img_dir_path=img_dir_path, 
		ishape=ishape, 
		total_batches=total_batches, 
		batch_size=batch_size)

	loss = []

	print('\nTrain epoch: {}'.format(epoch), end='')

	for batch_idx in range(total_batches):
		batchx4d, heatmap4d = next(gen)
		batch_loss = model.train_on_batch(batchx4d, heatmap4d)
		loss.append(batch_loss)

		if batch_idx%1 == 0:
			print('*', end='')
	
	mean_loss = float(np.mean(loss, axis=0))
	print(round(mean_loss, 2), end='')

	model.save_weights('{}/_weights.h5'.format(output_path))

	with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}, '.format(round(mean_loss, 2)))

	# Validate
	gen = genheatmap(
		anno_file_path=test_anno_file_path, 
		img_dir_path=test_img_dir_path, 
		ishape=ishape, 
		total_batches=total_test_batches, 
		batch_size=batch_size)

	loss = []

	print(', validate', end='')

	for batch_idx in range(total_test_batches):
		batchx4d, heatmap4d = next(gen)
		batch_loss = model.test_on_batch(batchx4d, heatmap4d)
		loss.append(batch_loss)

		if batch_idx%1 == 0:
			print('*', end='')

	mean_loss = float(np.mean(loss, axis=0))
	if mean_loss < min_loss:
		min_loss = mean_loss
		model.save_weights('{}/weights.h5'.format(output_path))

	print('{}/{}'.format(round(mean_loss, 2), round(min_loss, 5)), end='')

	with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}\n'.format(round(mean_loss, 2)))










