import tensorflow as tf
import numpy as np
from models import build_model
from datagen import load_dataset, genxy


net_name = "Net"
output_path = 'output'
anno_file_path = '/Users/baulhoa/Documents/PythonProjects/datasets/faceali/train.txt'
image_dir_path = '/Users/baulhoa/Documents/PythonProjects/datasets/faceali/train'
ishape = [112, 112, 1]
total_epoches = 1000
batch_size = 100

dataset = load_dataset(anno_file_path=anno_file_path)

total_examples = len(dataset)
total_batches = total_examples//batch_size

model = build_model(
	ishape=ishape, 
	mode='train', 
	net_name=net_name)
# model.summary()
# model.load_weights('{}/weights_.h5'.format(output_path), by_name=True)

min_loss = 2**32

for epoch in range(total_epoches):
	# tf.keras.backend.set_value(model.optimizer.learning_rate, 0.001)

	gen = genxy(
		dataset=dataset, 
		image_dir_path=image_dir_path, 
		ishape=ishape, 
		total_batches=total_batches, 
		batch_size=batch_size)

	print('\nTrain epoch {}'.format(epoch))
	loss = np.zeros(total_batches)

	for batch in range(total_batches):
		batchx4d, batchy4d = next(gen)
		batch_loss = model.train_on_batch(batchx4d, batchy4d)
		loss[batch] = batch_loss

		print('-', end='')
		if batch%100==99:
			print('{:.2f}%'.format((batch+1)*100/total_batches), end='\n')

	mean_loss = float(np.mean(loss, axis=-1))
	print('\nLoss: {:.3f}'.format(mean_loss))

	model.save_weights(output_path+'/weights_.h5')

