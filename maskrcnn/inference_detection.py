import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from models import build_train_maskrcnn_fpn
from utils import genanchors, box2frame, bbe2box2d, pnsm
from datagen import genx
from settings import settings


params = settings('fpn-inference')

start_example_index 		= params['start_example_index']
num_of_train_examples 		= params['num_of_train_examples']
num_of_validation_examples 	= params['num_of_validation_examples']
num_of_test_examples 		= params['num_of_test_examples']
asizes 						= params['asizes']
ishape 						= params['ishape']
ssizes 						= params['ssizes']
max_num_of_rois 			= params['max_num_of_rois']
unified_roi_size 			= params['unified_roi_size']
k0 							= params['k0']
top_down_pyramid_size 		= params['top_down_pyramid_size']
rpn_head_dim 				= params['rpn_head_dim']
fc_denses 					= params['fc_denses']
block_settings 				= params['resnet']
iou_thresholds 				= params['iou_thresholds']
nsm_iou_threshold			= params['nsm_iou_threshold']
nsm_score_threshold			= params['nsm_score_threshold']
classes 					= params['classes']
ann_file					= params['dataset_anno_file_path']
img_dir						= params['dataset_image_dir_path']
output_path					= params['output_path']

coco = COCO(ann_file)

lvl1_aboxes = genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0])
lvl2_aboxes = genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1])
lvl3_aboxes = genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2])
lvl4_aboxes = genanchors(isize=ishape[:2], ssize=ssizes[3], asizes=asizes[3])
anchor_4dtensors = [
	tf.constant(value=lvl1_aboxes, dtype='float32'), 
	tf.constant(value=lvl2_aboxes, dtype='float32'),
	tf.constant(value=lvl3_aboxes, dtype='float32'),
	tf.constant(value=lvl4_aboxes, dtype='float32')]

rpn_model, detection_model = build_train_maskrcnn_fpn(
	ishape=ishape, 
	anchor_4dtensors=anchor_4dtensors, 
	classes=classes, 
	max_num_of_rois=max_num_of_rois, 
	nsm_iou_threshold=nsm_iou_threshold, 
	nsm_score_threshold=nsm_score_threshold, 
	unified_roi_size=unified_roi_size,
	k0=k0,
	top_down_pyramid_size=top_down_pyramid_size,
	rpn_head_dim=rpn_head_dim,
	fc_denses=fc_denses,
	block_settings=block_settings,
	base_block_trainable=False)

print('Loads weights')
rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path))
detection_model.load_weights('{}/detection_weights.h5'.format(output_path))

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
		start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
	ishape=ishape)

for sample_order in range(num_of_test_examples):
	x, img_id = next(gen)

	# predict proposal
	batch_x = tf.expand_dims(input=x, axis=0)
	y_pred = rpn_model.predict_on_batch(batch_x)
	clzbbe_3dtensors = [y_pred[0][0], y_pred[1][0], y_pred[2][0], y_pred[3][0]]

	roi_2dtensor = pnsm(
		anchor_4dtensors=anchor_4dtensors, 
		clzbbe_3dtensors=clzbbe_3dtensors, 
		max_num_of_rois=max_num_of_rois,
		nsm_iou_threshold=nsm_iou_threshold,
		nsm_score_threshold=nsm_score_threshold,
		ishape=ishape)

	# predict object proposal
	batch_x = [tf.expand_dims(input=x, axis=0), tf.expand_dims(input=roi_2dtensor, axis=0)]
	y_pred = detection_model.predict_on_batch(batch_x)
	clzbbe_tensor = y_pred[0]
	clz_2dtensor = clzbbe_tensor[:, :len(classes)]
	bbe_2dtensor = clzbbe_tensor[:, len(classes):]
	pred_bbox2d = bbe2box2d(box_2dtensor=roi_2dtensor, bbe_2dtensor=bbe_2dtensor)

	_, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(x/255)
	ax.set_xlabel('Order: {}, Image ID: {}'.format(sample_order, img_id))
	
	for k in range(pred_bbox2d.shape[0]):
		clz = np.argmax(clz_2dtensor[k])
		
		# if clz == len(classes)-1:
		# 	continue

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
		
		ax.annotate(classes[clz], (tframe[0], tframe[1]), 
			color='green', weight='bold', fontsize=8, ha='center', va='center')

		tframe = box2frame(box=bbox, apoint=[0, 0])

		ax.add_patch(Rectangle(
			(tframe[0], tframe[1]), tframe[2], tframe[3],
			linewidth=1, 
			edgecolor='g',
			facecolor='none', 
			linestyle='-'))

	plt.show()











