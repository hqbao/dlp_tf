import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_infer_model
from datagen import load_dataset, genbbox
from utils import genanchors
from datetime import datetime


output_path = 'output'
test_anno_file_path = '../datasets/widerface/test.txt'
test_image_dir = '../datasets/widerface/test'
ishape = [1024, 1024, 3]
ssizes = [
	[256, 256],
	[128, 128],
	[64, 64], 
	[32, 32], 
]
asizes = [
	[[64, 64]],
	[[128, 128]],
	[[256, 256]],
	[[512, 512]],
]
total_classes = 1
resnet_settings = [[8, 8, 32], [8, [2, 2]], [8, [2, 2]], [8, [2, 2]], [4, [2, 2]]]
top_down_pyramid_size = 64
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 100
total_test_examples = 100

a1box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0]), dtype='float32') # (h1*w1*k1, 4)
a2box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1]), dtype='float32') # (h2*w2*k2, 4)
a3box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2]), dtype='float32') # (h3*w3*k3, 4)
a4box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[3], asizes=asizes[3]), dtype='float32') # (h4*w4*k4, 4)
abox_2dtensors = [a1box_2dtensor, a2box_2dtensor, a3box_2dtensor, a4box_2dtensor]
abox_2dtensor = tf.concat(values=abox_2dtensors, axis=0)

model = build_infer_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	top_down_pyramid_size=top_down_pyramid_size,
	k=[len(asizes[0]), len(asizes[1]), len(asizes[2]), len(asizes[3])], 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_recall.h5'.format(output_path), by_name=True)

test_dataset = load_dataset(anno_file_path=test_anno_file_path)
gen = genbbox(
	dataset=test_dataset, 
	image_dir=test_image_dir, 
	ishape=ishape,
	total_examples=total_test_examples)

for batch in range(total_test_examples):
	image, _ = next(gen)
	batchx_4dtensor = tf.constant(value=[image], dtype='float32')
	print('{}: 1'.format(datetime.now().time()), end='\n')
	boxclz_2dtensor, valid_outputs = model.predict_on_batch(batchx_4dtensor) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, total_classes+1+4)
	print('{}: 2'.format(datetime.now().time()), end='\n')
	total_boxes = valid_outputs[0].numpy()
	if total_boxes > 0:
		boxclz_2dtensor = boxclz_2dtensor[:total_boxes]
		pix = np.array(batchx_4dtensor[0], dtype='uint8')

		_, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(pix)

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














