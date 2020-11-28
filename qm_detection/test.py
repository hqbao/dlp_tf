import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_infer_model
from datagen import genx
from utils import genanchors
from datetime import datetime


test_anno_file_path = '../datasets/quizmarker/qm/test_anno.txt'
test_image_dir = '../datasets/quizmarker/qm/full_images'
ishape = [240, 200, 3]
ssize = [60, 50]
asizes = [[8, 8]]
resnet_settings = [[5, 5, 20], [2, [1, 1]], [8, [2, 2]]]
total_classes = 2
output_path = 'output'
nsm_iou_threshold = 0.1
nsm_score_threshold = 0.9
nsm_max_output_size = 500
total_test_examples = 100

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
abox_4dtensor = tf.constant(value=abox4d, dtype='float32')
abox_2dtensor = tf.reshape(tensor=abox_4dtensor, shape=[-1, 4])

model = build_infer_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
model.summary()
# model.load_weights('{}/weights.h5'.format(output_path), by_name=True)

gen = genx(
	anno_file_path=test_anno_file_path, 
	image_dir=test_image_dir, 
	ishape=ishape,
	mode='test')

for batch in range(total_test_examples):
	batchx_4dtensor, _, image_id = next(gen)
	print('{}: 1'.format(datetime.now().time()), end='\n')
	boxclz_2dtensor, valid_outputs = model.predict_on_batch(batchx_4dtensor) # (h*w*k, total_classes+1+4)
	print('{}: 2'.format(datetime.now().time()), end='\n')
	image = batchx_4dtensor[0]

	_, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(image)
	ax.set_xlabel('Image ID: {}'.format(image_id))

	for i in range(valid_outputs[0]):
		box = boxclz_2dtensor[i, :4]
		clz = boxclz_2dtensor[i, 4]

		color = 'cyan' if clz == 0 else 'yellow'
		frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3],
			linewidth=0.5, 
			edgecolor=color,
			facecolor='none', 
			linestyle='-'))

	plt.show()














