import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from model import net
from vggface2_datagen import genxyclz


img_dir_path = '../datasets/vggface2/train_refined_resized/'
anno_file_path = 'anno/vggface2_test1_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
total_classes = 1000
total_examples = 10000
batch_size = 1
total_batches = total_examples//batch_size

input_tensor = Input(shape=ishape)
_, [t1, t2, t3, t4, t5] = net(ishape=ishape, input_tensor=input_tensor, trainable=False)
model = Model(inputs=input_tensor, outputs=[t1, t2, t3, t4, t5])
model.load_weights('{}/clz/weights.h5'.format(output_path), by_name=True)

gen = genxyclz(
	anno_file_path=anno_file_path, 
	img_dir_path=img_dir_path, 
	ishape=ishape, 
	total_examples=total_examples, 
	batch_size=batch_size)

for _ in range(total_batches):
	batchx4d, id_batchy1d, = next(gen)

	# if id_batchy1d[0] != 0:
	# 	continue

	x = batchx4d[0]
	prediction = model.predict_on_batch(batchx4d)

	t1_3dtensor = prediction[0][0]
	t2_3dtensor = prediction[1][0]
	t3_3dtensor = prediction[2][0]
	t4_3dtensor = prediction[3][0]
	t5_3dtensor = prediction[4][0]

	t1_2dtensor = tf.math.reduce_mean(input_tensor=t1_3dtensor, axis=2)
	t2_2dtensor = tf.math.reduce_mean(input_tensor=t2_3dtensor, axis=2)
	t3_2dtensor = tf.math.reduce_mean(input_tensor=t3_3dtensor, axis=2)
	t4_2dtensor = tf.math.reduce_mean(input_tensor=t4_3dtensor, axis=2)
	t5_2dtensor = tf.math.reduce_mean(input_tensor=t5_3dtensor, axis=2)

	t12345 = [t1_2dtensor, t2_2dtensor, t3_2dtensor, t4_2dtensor, t5_2dtensor]

	_, ax = plt.subplots(2, 3, figsize=(15, 7.35))

	ax[0, 0].imshow(x/255)

	for lvl in range(1, 6):
		x = t12345[lvl-1]
		ax[int(lvl/3), lvl%3].imshow(x)
		ax[int(lvl/3), lvl%3].set_xlim([0, x.shape[1]])
		ax[int(lvl/3), lvl%3].set_ylim([x.shape[0], 0])

	plt.show()


	








