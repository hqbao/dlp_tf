import numpy as np
import tensorflow as tf
from datetime import datetime
from vggface2_datagen import gentriplets
from model import build_ver_model, build_inf_model


img_dir_path = '../datasets/vggface2/train_refined_resized/'
anno_file_path = 'anno/vggface2_train3_anno.txt'
test_anno_file_path = 'anno/vggface2_test3_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [1, 1]]]
dense_settings = []
embedding_dims = 512
total_epoches = 1000
total_identities = 10
total_images = 20
total_test_images = 20
total_examples = 8550
total_test_examples = 90
batch_size = 100
total_batches = total_examples//batch_size
alpha = 0.2

log_file_name = "{}:train-ver-log.txt".format(datetime.now().time())
max_acc = 0

model = build_ver_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings, 
	embedding_dims=embedding_dims, alpha=alpha)
test_model = build_inf_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings, 
	embedding_dims=embedding_dims)
model.summary()
# test_model.summary()
model.load_weights('{}/clz/weights.h5'.format(output_path), by_name=True)

for epoch in range(total_epoches):
	gen = gentriplets(
		anno_file_path=anno_file_path,
		img_dir_path=img_dir_path,
		ishape=ishape,
		total_identities=total_identities,
		total_images=total_images,
		total_examples=total_examples,
		batch_size=batch_size)

	loss = []

	for batch_idx in range(total_batches):
		batchx4d, _ = next(gen)
		batchy2d = np.zeros((3*batch_size, total_identities), dtype='float32')
		batch_loss = model.train_on_batch(batchx4d, batchy2d)
		loss.append(batch_loss)

		if batch_idx%10 == 0:
			print('-', end='')

	model.save_weights('{}/ver/_weights.h5'.format(output_path))

	mean_loss = np.mean(loss, axis=0)
	print(round(mean_loss, 2))

	with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}, '.format(round(mean_loss, 2)))
  
	test_model.load_weights('{}/ver/_weights.h5'.format(output_path), by_name=True)

	batchx4d, batchy1d = genpairs(
		anno_file_path=test_anno_file_path,
		img_dir_path=img_dir_path,
		ishape=ishape,
		total_identities=total_identities,
		total_images=total_test_images,
		total_examples=total_test_examples,
		difference_rate=0.5)

	pred_batchy1d = test_model.predict_on_batch(batchx4d)

	threshold = tf.reduce_mean(input_tensor=pred_batchy1d, axis=-1).numpy()
	print('Threshold: {}'.format(round(threshold, 2)))

	batchy_1dtensor = tf.constant(value=batchy1d, dtype='bool')

	pred_batchy1d = tf.where(
		condition=tf.math.greater(
			x=pred_batchy1d,
			y=threshold),
		x=False,
		y=True)

	true_positives = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.equal(x=batchy_1dtensor, y=True),
			y=tf.math.equal(x=pred_batchy1d, y=True)),
		x=1,
		y=0)

	true_negatives = tf.where(
		condition=tf.math.logical_and(
			x=tf.math.equal(x=batchy_1dtensor, y=False),
			y=tf.math.equal(x=pred_batchy1d, y=False)),
		x=1,
		y=0)

	true_positives_count = tf.reduce_sum(input_tensor=true_positives).numpy()
	true_negatives_count = tf.reduce_sum(input_tensor=true_negatives).numpy()
	accuracy = (true_positives_count+true_negatives_count)/batchy_1dtensor.shape[0]

	print('Accuracy: {}'.format(round(accuracy, 4)))

	if max_acc < accuracy:
		max_acc = accuracy
		model.save_weights('{}/ver/weights.h5'.format(output_path))








