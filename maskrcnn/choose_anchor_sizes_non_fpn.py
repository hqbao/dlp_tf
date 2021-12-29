import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from datagen import genx, gety
from utils import genanchors, comiou4d


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

start_example_index = 0
num_of_examples = 2000

asizes = [[91, 181], [128, 128], [181, 91]]
ishape = [1024, 1024, 3]
ssize = [32, 32]
frame_mode = True
classes = ['face', 'none']
mapping = {0: 0}
iou_thresholds = [0.3, 0.5]
ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
anchor_4dtensor = tf.constant(value=abox4d, dtype='float32')

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=[start_example_index, start_example_index+num_of_examples],
	ishape=ishape)

count = 0
not_coverable = []

for sample_order in range(num_of_examples):

	# generate x
	x, img_id = next(gen)

	# get labels
	bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

	H = anchor_4dtensor.shape[0]
	W = anchor_4dtensor.shape[1]
	K = anchor_4dtensor.shape[2]
	num_of_bboxes = bbox2d.shape[0]

	matching = 0

	for i in range(num_of_bboxes):
		bbox_1dtensor = tf.constant(value=bbox2d[i], dtype='float32')
		bbox_2dtensor = tf.expand_dims(input=bbox_1dtensor, axis=0)
		bbox_3dtensor = tf.expand_dims(input=bbox_2dtensor, axis=0)
		bbox_4dtensor = tf.expand_dims(input=bbox_3dtensor, axis=0)

		iou_4dtensor = comiou4d(bbox_4dtensor=bbox_4dtensor, anchor_4dtensor=anchor_4dtensor) # (h, w, k, 1)
		max_iou_tensor = tf.math.reduce_max(input_tensor=iou_4dtensor, axis=None)
		max_iou = max_iou_tensor.numpy()

		matching += int(max_iou >= iou_thresholds[1])

	if matching == num_of_bboxes:
		count += 1
	else:
		not_coverable.append(img_id)

	print('-', end='')
	if sample_order%100==99:
		print('{}%'.format(round(sample_order*100/num_of_examples, 2)))	

print('\n{}/{}'.format(count, num_of_examples))
for img_id in not_coverable:
	print(img_id, end=',')

		







