import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from datagen import genx, gety
from utils import genanchors, box2frame, comiou4d


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

start_example_index = 0
num_of_examples = 2181

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

for sample_order in range(num_of_examples):

	# generate x
	x, img_id = next(gen)

	# get labels
	bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

	_, ax = plt.subplots(figsize=(15, 7.35))

	H = anchor_4dtensor.shape[0]
	W = anchor_4dtensor.shape[1]
	K = anchor_4dtensor.shape[2]
	num_of_bboxes = bbox2d.shape[0]

	# Creates bbox_4dtensors
	bbox5d = np.zeros((num_of_bboxes, H, W, K, 4), dtype='float32')
	for i in range(num_of_bboxes):
		bbox5d[i, :, :, :, 0] = bbox2d[i, 0]
		bbox5d[i, :, :, :, 1] = bbox2d[i, 1]
		bbox5d[i, :, :, :, 2] = bbox2d[i, 2]
		bbox5d[i, :, :, :, 3] = bbox2d[i, 3]

	bbox_4dtensors = []
	for i in range(num_of_bboxes):
		bbox_4dtensors.append(tf.constant(value=bbox5d[i], dtype='float32'))

	# Computes iou_tensors
	iou_4dtensors = []
	for i in range(num_of_bboxes):
		iou_4dtensor = comiou4d(bbox_4dtensor=bbox_4dtensors[i], anchor_4dtensor=anchor_4dtensor) # (h, w, k, 1)
		iou_4dtensors.append(iou_4dtensor)

	# Assigns positive, neutral, negative labels to anchors
	zero_4dtensor = tf.zeros(shape=(H, W, K, 1), dtype='float32') # (h, w, k, 1)
	one_4dtensor = tf.ones(shape=(H, W, K, 1), dtype='float32') # (h, w, k, 1)
	iou_threshold_tensor = tf.constant(value=iou_thresholds, dtype='float32') # (2,)

	pos_5dtensors = []
	neg_5dtensors = []
	for i in range(num_of_bboxes):

		if False:
			# Finds max iou and assign it to zero tensor
			iou_3dtensor = tf.math.reduce_max(input_tensor=iou_4dtensors[i], axis=0)
			iou_2dtensor = tf.math.reduce_max(input_tensor=iou_3dtensor, axis=0)
			iou_1dtensor = tf.math.reduce_max(input_tensor=iou_2dtensor, axis=0)
			max_iou_idx = tf.math.argmax(input=iou_1dtensor, axis=0, output_type='int32')
			max_iou = iou_1dtensor[max_iou_idx]
			zero_max_4dtensor = tf.where(
				condition=tf.math.equal(x=iou_4dtensors[i], y=max_iou),
				x=one_4dtensor,
				y=zero_4dtensor)
		else:
			zero_max_4dtensor = zero_4dtensor

		# Assign 1 to positive anchors
		pos_4dtensor = tf.where(
			condition=tf.math.greater_equal(x=iou_4dtensors[i], y=iou_threshold_tensor[1]),
			x=one_4dtensor,
			y=zero_max_4dtensor) # (h, w, k, 1)
		pos_5dtensor = tf.expand_dims(input=pos_4dtensor, axis=0) # (1, h, w, k, 1)
		pos_5dtensors.append(pos_5dtensor) 

		# Assign -1 to negative anchors
		neg_4dtensor = tf.where(
			condition=tf.math.less_equal(x=iou_4dtensors[i], y=iou_threshold_tensor[0]),
			x=-one_4dtensor,
			y=zero_4dtensor) # (h, w, k, 1)
		neg_5dtensor = tf.expand_dims(input=neg_4dtensor, axis=0) # (1, h, w, k, 1)
		neg_5dtensors.append(neg_5dtensor) 

	pos_5dtensor = tf.concat(values=pos_5dtensors, axis=0) # (num_of_bboxes, h, w, k, 1)
	neg_5dtensor = tf.concat(values=neg_5dtensors, axis=0) # (num_of_bboxes, h, w, k, 1)

	sum_pos_4dtensor = tf.reduce_sum(input_tensor=pos_5dtensor, axis=0) # (h, w, k, 1)
	sum_neg_4dtensor = tf.reduce_sum(input_tensor=neg_5dtensor, axis=0) # (h, w, k, 1)

	# Shows
	sum_pos_4dtensor = np.array(sum_pos_4dtensor)
	sum_neg_4dtensor = np.array(sum_neg_4dtensor)

	ax.imshow(x/255)
	ax.set_xlabel('Order: {}, Image ID: {}'.format(sample_order, img_id))

	for h in range(H):
		for w in range(W):
			for k in range(K):

				frame = box2frame(box=abox4d[h, w, k], apoint=[0, 0])

				# positive anchors
				if sum_pos_4dtensor[h, w, k, 0] > 0.0:
					if True: 
						ax.add_patch(Rectangle(
							(frame[0], frame[1]), frame[2], frame[3],
							linewidth=1, 
							edgecolor='cyan',
							facecolor='none', 
							linestyle='-'))

				# negative anchors
				if sum_neg_4dtensor[h, w, k, 0] == -num_of_bboxes:
					if False: 
						ax.add_patch(Rectangle(
							(frame[0], frame[1]), frame[2], frame[3],
							linewidth=0.1, 
							edgecolor='red',
							facecolor='none', 
							linestyle='-'))

				# neutral anchors
				if sum_pos_4dtensor[h, w, k, 0] <= 0 and sum_neg_4dtensor[h, w, k, 0] > -num_of_bboxes:
					if True: 
						ax.add_patch(Rectangle(
							(frame[0], frame[1]), frame[2], frame[3],
							linewidth=0.2, 
							edgecolor='white',
							facecolor='none', 
							linestyle='-'))

	plt.show()







