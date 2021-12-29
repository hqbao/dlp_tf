import numpy as np
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from models import build_train_maskrcnn_non_fpn
from utils import genanchors, pad_roi2d, bbe2box2d, box2frame
from datagen import genx, gety, gendy


start_example_index = 0
num_of_train_examples = 1
num_of_validation_examples = 0
num_of_test_examples = 100

asizes = [[91, 181], [128, 128], [181, 91]]
ishape = [1024, 1024, 3]
ssize = [32, 32]
frame_mode = True
classes = ['face', 'none']
mapping = {0: 0}
iou_thresholds = [0.3, 0.5]
max_num_of_rois = 7
unified_roi_size = [7, 7]
rpn_head_dim = 64
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.1
fc_denses = [1024, 1024]
block_settings = [[16, 16, 64], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [1, 1]]]
output_path = 'output'
ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

##################################
# DETECTION TRAIN & VALIDATION
##################################

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
anchor_4dtensor = tf.constant(value=abox4d, dtype='float32')

rpn_model, detection_model = build_train_maskrcnn_non_fpn(
	ishape=ishape, 
	anchor_4dtensor=anchor_4dtensor, 
	classes=classes, 
	max_num_of_rois=max_num_of_rois, 
	nsm_iou_threshold=nsm_iou_threshold, 
	nsm_score_threshold=nsm_score_threshold, 
	unified_roi_size=unified_roi_size,
	rpn_head_dim=rpn_head_dim,
	fc_denses=fc_denses,
	block_settings=block_settings,
	base_block_trainable=True)

# load weihts
# detection_model.load_weights('{}/detection_weights.h5'.format(output_path))

if True:

	for epoch in range(1):

		gen = genx(
			coco=coco, 
			img_dir=img_dir, 
			classes=classes, 
			limit=(start_example_index, 
				start_example_index+num_of_train_examples+num_of_validation_examples),
			ishape=ishape)

		#------------------------------
		# TRAIN
		#------------------------------

		print('{}: TRAIN {}'.format(datetime.now().time(), epoch), end='\n')

		loss = np.zeros((num_of_train_examples, 1))

		for batch_idx in range(num_of_train_examples):

			# generate x
			x, img_id = next(gen)

			# get labels
			bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

			# predict rois
			roi2d = pad_roi2d(roi2d=bbox2d[:, :4], max_num_of_rois=max_num_of_rois, pad_roi=[0, 0, ishape[0], ishape[1]])
			roi_2dtensor = tf.constant(value=roi2d, dtype='float32')

			clzbbe_2dtensor = gendy(
				num_of_classes=len(classes), 
				roi_2dtensor=roi_2dtensor, 
				bbox2d=bbox2d, 
				unified_roi_size=unified_roi_size,
				isize=ishape[:2], 
				iou_thresholds=iou_thresholds)

			batch_x = [tf.expand_dims(input=x, axis=0), tf.expand_dims(input=roi_2dtensor, axis=0)]
			batch_y = tf.expand_dims(input=clzbbe_2dtensor, axis=0)

			batch_loss = detection_model.train_on_batch(batch_x, batch_y)
			loss[batch_idx, :] = batch_loss

			print('-', end='')
			if batch_idx%100==99:
				print('{}%'.format(round(batch_idx*100/num_of_train_examples, 2)))

		print()
		print(np.mean(loss, axis=0))

		detection_model.save_weights('{}/detection_weights.h5'.format(output_path))


if True:

	gen = genx(
		coco=coco, 
		img_dir=img_dir, 
		classes=classes, 
		limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
			start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
		ishape=ishape)

	for batch_idx in range(num_of_test_examples):

		# generate x
		x, img_id = next(gen)

		# get labels
		bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

		# predict rois
		roi2d = pad_roi2d(roi2d=bbox2d[:, :4], max_num_of_rois=max_num_of_rois, pad_roi=[0, 0, ishape[0], ishape[1]])
		roi_2dtensor = tf.constant(value=roi2d, dtype='float32')

		# predict object proposal
		batch_x = [tf.expand_dims(input=x, axis=0), tf.expand_dims(input=roi_2dtensor, axis=0)]
		y_pred = detection_model.predict_on_batch(batch_x)
		clzbbe_tensor = y_pred[0]
		clz_2dtensor = clzbbe_tensor[:, :len(classes)]
		bbe_2dtensor = clzbbe_tensor[:, len(classes):]
		pred_bbox2d = bbe2box2d(box_2dtensor=roi_2dtensor, bbe_2dtensor=bbe_2dtensor)

		_, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(x/255)
		
		# iterate over rois
		for k in range(pred_bbox2d.shape[0]):

			print(np.array(clz_2dtensor[k]))
			# print('{} -> {}'.format(k, classes[clz]))

			clz = np.argmax(clz_2dtensor[k])
			roi = roi_2dtensor[k]
			bbox = pred_bbox2d[k]

			rframe = box2frame(box=roi, apoint=[0, 0])

			ax.add_patch(Rectangle(
				(rframe[0], rframe[1]), rframe[2], rframe[3], 
				linewidth=1, 
				edgecolor='r',
				facecolor='none', 
				linestyle='-'))

			tframe = box2frame(box=bbox, apoint=[0.5, 0.5])
			
			ax.annotate(classes[clz], (tframe[0], tframe[1]), color='green', weight='bold', 
		            fontsize=8, ha='center', va='center')

			tframe = box2frame(box=bbox, apoint=[0, 0])

			ax.add_patch(Rectangle(
				(tframe[0], tframe[1]), tframe[2], tframe[3],
				linewidth=1, 
				edgecolor='g',
				facecolor='none', 
				linestyle='-'))

		plt.show()







