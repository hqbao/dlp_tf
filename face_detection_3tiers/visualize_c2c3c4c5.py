import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from datagen import genx
from models import resnet


mode = 'train'
anno_file_path = '../datasets/widerface/full_'+mode+'_anno.txt'
image_dir = '../datasets/widerface/full_'+mode+'_images'
output_path = 'output'
ishape = [256, 256, 3]
total_examples = 10593

use_bias = True
weight_decay = 0.001
trainable = False
bn_trainable = False
resnet_settings = [[32, 32, 128], [3, [1, 1]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]]

input_tensor = Input(shape=ishape)
C2, C3, C4, C5 = resnet(
	input_tensor=input_tensor, 
	block_settings=resnet_settings, 
	use_bias=use_bias, 
	weight_decay=weight_decay,
	trainable=trainable,
	bn_trainable=bn_trainable)
model = Model(inputs=input_tensor, outputs=[C2, C3, C4, C5])

model.load_weights('{}/_weights.h5'.format(output_path), by_name=True)

gen = genx(
	anno_file_path=anno_file_path, 
	image_dir=image_dir, 
	ishape=ishape)

for _ in range(total_examples):
	x, _ = next(gen)
	prediction = model.predict_on_batch(x)
	x = x[0]

	c2_3dtensor = prediction[0][0]
	c3_3dtensor = prediction[1][0]
	c4_3dtensor = prediction[2][0]
	c5_3dtensor = prediction[3][0]

	c2_2dtensor = tf.math.reduce_mean(input_tensor=c2_3dtensor, axis=2)
	c3_2dtensor = tf.math.reduce_mean(input_tensor=c3_3dtensor, axis=2)
	c4_2dtensor = tf.math.reduce_mean(input_tensor=c4_3dtensor, axis=2)
	c5_2dtensor = tf.math.reduce_mean(input_tensor=c5_3dtensor, axis=2)

	c12345 = [c2_2dtensor, c3_2dtensor, c4_2dtensor, c5_2dtensor]

	_, ax = plt.subplots(2, 3, figsize=(15, 7.35))

	ax[0, 0].imshow(x/255)

	for lvl in range(1, 5):
		x = c12345[lvl-1]
		ax[int(lvl/3), lvl%3].imshow(x)
		ax[int(lvl/3), lvl%3].set_xlim([0, x.shape[1]])
		ax[int(lvl/3), lvl%3].set_ylim([x.shape[0], 0])

	plt.show()


	








