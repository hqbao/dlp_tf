import tensorflow as tf
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from net_blocks import resnet
from datagen import genx


ishape = [1024, 1024, 3]
output_path = 'output'
classes = ['face', 'none']
start_example_index = 0
num_of_examples = 100

input_tensor = Input(shape=ishape)

block_settings = [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]] # Resnet 50, pool 64
# block_settings = [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [23, [1, 1]], [3, [1, 1]]] # Resnet 101, pool 16
# block_settings = [[64, 64, 256], [3, [2, 2]], [8, [2, 2]], [36, [2, 2]], [3, [2, 2]]] # Resnet 152, pool 64
C1, C2, C3, C4, C5 = resnet(input_tensor=input_tensor, block_settings=block_settings, trainable=False)
model = Model(inputs=input_tensor, outputs=[C1, C2, C3, C4, C5])

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

	c1_3dtensor = prediction[0][0]
	c2_3dtensor = prediction[1][0]
	c3_3dtensor = prediction[2][0]
	c4_3dtensor = prediction[3][0]
	c5_3dtensor = prediction[4][0]

	c1_2dtensor = tf.math.reduce_mean(input_tensor=c1_3dtensor, axis=2)
	c2_2dtensor = tf.math.reduce_mean(input_tensor=c2_3dtensor, axis=2)
	c3_2dtensor = tf.math.reduce_mean(input_tensor=c3_3dtensor, axis=2)
	c4_2dtensor = tf.math.reduce_mean(input_tensor=c4_3dtensor, axis=2)
	c5_2dtensor = tf.math.reduce_mean(input_tensor=c5_3dtensor, axis=2)

	c12345 = [c1_2dtensor, c2_2dtensor, c3_2dtensor, c4_2dtensor, c5_2dtensor]

	_, ax = plt.subplots(2, 3, figsize=(15, 7.35))

	ax[0, 0].imshow(x/255)

	for lvl in range(1, 6):
		x = c12345[lvl-1]
		ax[int(lvl/3), lvl%3].imshow(x)
		ax[int(lvl/3), lvl%3].set_xlim([0, x.shape[1]])
		ax[int(lvl/3), lvl%3].set_ylim([x.shape[0], 0])

	plt.show()


	








