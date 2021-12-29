import tensorflow as tf
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from net_blocks import fpn
from datagen import genx


ishape = [1024, 1024, 3]
output_path = 'output'
classes = ['face', 'none']
start_example_index = 0
num_of_examples = 100
block_settings = [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]] # Resnet 50, pool 8
top_down_pyramid_size = 512


input_tensor = Input(shape=ishape)

P2, P3, P4, P5 = fpn(
	input_tensor=input_tensor, 
	block_settings=block_settings, 
	top_down_pyramid_size=top_down_pyramid_size, 
	trainable=False)

model = Model(inputs=input_tensor, outputs=[P2, P3, P4, P5])

model.load_weights('{}/rpn_weights.h5'.format(output_path), by_name=True)

ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

gen = genx(
	coco=coco,
	img_dir=img_dir,
	classes=classes, 
	limit=[start_example_index, start_example_index+num_of_examples],
	ishape=ishape)

for _ in range(num_of_examples):
	x, _ = next(gen)
	batch_x = tf.expand_dims(input=x, axis=0)

	prediction = model.predict_on_batch(batch_x)

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


	








