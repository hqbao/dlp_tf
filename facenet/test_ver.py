import skimage.io as io
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from vggface2_datagen import genpairs
from model import build_inf_model


img_dir_path = '../datasets/vggface2/train_refined_resized/'
test_anno_file_path = 'anno/vggface2_test2_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [1, 1]]]
dense_settings = []
embedding_dims = 1024
total_identities = 33
total_test_images = 20
total_test_examples = 1000

test_model = build_inf_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings,
	embedding_dims=embedding_dims)
# test_model.summary()
test_model.load_weights('{}/clz/weights.h5'.format(output_path), by_name=True)

batchx4d, batchy1d = genpairs(
	anno_file_path=test_anno_file_path,
	img_dir_path=img_dir_path,
	ishape=ishape,
	total_identities=total_identities,
	total_images=total_test_images,
	total_examples=total_test_examples,
	difference_rate=0.5)

pred_batchy1d = test_model.predict_on_batch(batchx4d)

# print(pred_batchy1d)
threshold = tf.reduce_mean(input_tensor=pred_batchy1d, axis=-1).numpy()
print('Threshold: {}'.format(threshold))

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

true_positives_count = tf.reduce_sum(input_tensor=true_positives)
true_negatives_count = tf.reduce_sum(input_tensor=true_negatives)
accuracy = (true_positives_count+true_negatives_count)/batchy_1dtensor.shape[0]

print('Accuracy: {}'.format(accuracy))

# for exp_idx in range(total_test_examples):
# 	fig, ax = plt.subplots(1, 2, figsize=(15, 7.35))
# 	fig.suptitle('{} -> {}'.format(bool(batchy1d[exp_idx]), pred_batchy1d[exp_idx]), fontsize=16)
# 	ax[0].imshow(batchx4d[2*exp_idx]/255)
# 	ax[1].imshow(batchx4d[2*exp_idx+1]/255)
# 	plt.show()






