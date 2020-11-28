import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_infer_model
from datagen import genbbox
from utils import genanchors
from datetime import datetime


test_anno_file_path = '../datasets/widerface/test.txt'
test_image_dir = '../datasets/widerface/test'
output_path = 'output'
ishape = [256, 256, 3]
ssizes = [
	[64, 64], 
	[32, 32], 
	[16, 16],
]
asizes = [
	[[32, 32]],
	[[64, 64]],
	[[128, 128]],
]
total_classes = 1
resnet_settings = [[16, 16, 64], [4, [2, 2]], [8, [2, 2]], [16, [2, 2]]]
top_down_pyramid_size = 64
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 1000
total_test_examples = 1000

a1box2d = genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0]) # (h1 * w1 * k1, 4)
a2box2d = genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1]) # (h2 * w2 * k2, 4)
a3box2d = genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2]) # (h3 * w3 * k3, 4)
abox2d = np.concatenate([a1box2d, a2box2d, a3box2d], axis=0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, 4)
abox_2dtensor = tf.constant(value=abox2d, dtype='float32') # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, 4)

model = build_infer_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	top_down_pyramid_size=top_down_pyramid_size,
	k=[len(asizes[0]), len(asizes[1]), len(asizes[2])], 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_recall.h5'.format(output_path), by_name=True)

gen = genbbox(
	anno_file_path=test_anno_file_path, 
	image_dir=test_image_dir, 
	ishape=ishape)

for batch in range(total_test_examples):
	image, _, image_id = next(gen)
	batchx_4dtensor = tf.constant(value=[image], dtype='int32')
	print('{}: 1'.format(datetime.now().time()), end='\n')
	boxclz_2dtensor, valid_outputs = model.predict_on_batch(batchx_4dtensor) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1+4)
	print('{}: 2'.format(datetime.now().time()), end='\n')
	total_boxes = valid_outputs[0].numpy()
	if total_boxes > 0:
		boxclz_2dtensor = boxclz_2dtensor[:total_boxes]
		x = batchx_4dtensor[0]

		_, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(x)
		ax.set_xlabel('Image ID: {}'.format(image_id))

		for i in range(boxclz_2dtensor.shape[0]):
			box = boxclz_2dtensor[i, :4]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.5, 
				edgecolor='cyan',
				facecolor='none', 
				linestyle='-'))

		plt.show()














