import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from datagen import genx
from models import fpn


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
top_down_pyramid_size = 256


input_tensor = Input(shape=ishape)
P2, P3, P4, P5 = fpn(
	input_tensor=input_tensor, 
	resnet_settings=resnet_settings, 
	top_down_pyramid_size=top_down_pyramid_size, 
	use_bias=use_bias,
	weight_decay=weight_decay,
	trainable=trainable,
	bn_trainable=bn_trainable)

model = Model(inputs=input_tensor, outputs=[P2, P3, P4, P5])

model.load_weights('{}/_weights.h5'.format(output_path), by_name=True)

gen = genx(
	anno_file_path=anno_file_path, 
	image_dir=image_dir, 
	ishape=ishape)

for _ in range(total_examples):
	x, _ = next(gen)
	prediction = model.predict_on_batch(x)
	x = x[0]

	p2_3dtensor = prediction[0][0]
	p3_3dtensor = prediction[1][0]
	p4_3dtensor = prediction[2][0]
	p5_3dtensor = prediction[3][0]

	p2_2dtensor = tf.math.reduce_mean(input_tensor=p2_3dtensor, axis=2)
	p3_2dtensor = tf.math.reduce_mean(input_tensor=p3_3dtensor, axis=2)
	p4_2dtensor = tf.math.reduce_mean(input_tensor=p4_3dtensor, axis=2)
	p5_2dtensor = tf.math.reduce_mean(input_tensor=p5_3dtensor, axis=2)

	p2345 = [p2_2dtensor, p3_2dtensor, p4_2dtensor, p5_2dtensor]

	_, ax = plt.subplots(2, 3, figsize=(15, 7.35))

	ax[0, 0].imshow(x/255)

	for lvl in range(1, 5):
		x = p2345[lvl-1]
		ax[int(lvl/3), lvl%3].imshow(x/255)
		ax[int(lvl/3), lvl%3].set_xlim([0, x.shape[1]])
		ax[int(lvl/3), lvl%3].set_ylim([x.shape[0], 0])

	plt.show()


	








