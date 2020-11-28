import numpy as np
import tensorflow as tf
from datetime import datetime
from vggface2_datagen import genxyclz
from model import build_clz_model


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

img_dir_path = '../datasets/vggface2/train_refined_resized/'
anno_file_path = 'anno/vggface2_test1_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[16, 16, 64], [2, [2, 2]], [2, [2, 2]], [2, [2, 2]]]
dense_settings = []
total_identities = 3000
total_examples = 60000
batch_size = 1000
total_batches = total_examples//batch_size

model = build_clz_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings, 
	total_classes=total_identities)
# model.summary()
model.load_weights('{}/clz/weights1.h5'.format(output_path), by_name=True)

gen = genxyclz(
	anno_file_path=anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_batches=total_batches, 
	batch_size=batch_size, 
	total_classes=total_identities)

for _ in range(total_batches):
	batchx4d, batchy2d = next(gen)
	batchy_2dtensor = tf.constant(value=batchy2d, dtype='int64') # (batch_size, total_identities)
	batchy_1dtensor = tf.math.argmax(input=batchy_2dtensor, axis=-1) # (batch_size,)

	print('Start: {}'.format(datetime.now().time()))
	pred_batchy_2dtensor = model.predict_on_batch(batchx4d) # (batch_size, total_identities)
	print('End: {}'.format(datetime.now().time()))
	pred_batchy_1dtensor = tf.math.argmax(input=pred_batchy_2dtensor, axis=-1) # (batch_size,)

	# Identity
	true_positive_1dtensor = tf.where(
		condition=tf.math.equal(x=batchy_1dtensor, y=pred_batchy_1dtensor),
		x=1,
		y=0)
	false_positive_1dtensor = tf.where(
		condition=tf.math.not_equal(x=batchy_1dtensor, y=pred_batchy_1dtensor),
		x=1,
		y=0)
	true_positives = tf.math.reduce_sum(input_tensor=true_positive_1dtensor, axis=-1).numpy()
	false_positives = tf.math.reduce_sum(input_tensor=false_positive_1dtensor, axis=-1).numpy()
	print('Identity precision: {}, correct: {}/{}'.format(true_positives/(true_positives+false_positives), true_positives, pred_batchy_1dtensor.shape[0]))





