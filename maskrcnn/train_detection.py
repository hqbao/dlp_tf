import numpy as np
from datetime import datetime
import tensorflow as tf

from pycocotools.coco import COCO
from models import build_train_maskrcnn_fpn
from utils import genanchors, pnsm
from datagen import genx, gety, gendy
from settings import settings


params = settings('fpn-train')

start_example_index 		= params['start_example_index']
num_of_train_examples 		= params['num_of_train_examples']
num_of_validation_examples 	= params['num_of_validation_examples']
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
base_block_trainable 		= params['base_block_trainable']
classes 					= params['classes']
mapping 					= params['mapping']
frame_mode 					= params['frame_mode']
ann_file					= params['dataset_anno_file_path']
img_dir						= params['dataset_image_dir_path']
output_path					= params['output_path']
weight_loading 				= params['weight_loading']
num_of_epoches 				= params['num_of_epoches']

coco = COCO(ann_file)

log_file_name = "{}:detection-train-log.txt".format(datetime.now().time())

##################################
# DETECTION TRAIN & VALIDATION
##################################

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
	base_block_trainable=base_block_trainable)

with open('{}/{}'.format(output_path, log_file_name), 'w') as log:
    detection_model.summary(print_fn=lambda x: log.write(x + '\n'))

# load weihts
if weight_loading is True:
	print('Loads weights')
	rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path))
	# detection_model.load_weights('{}/detection_weights.h5'.format(output_path))

for epoch in range(num_of_epoches):

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
		bbox2d, tmasks = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

		# predict rois
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

	with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}: epoch: {}, det loss: {}\n'.format(datetime.now().time(), epoch, np.mean(loss, axis=0)))

	rpn_model.save_weights('{}/rpn_weights.h5'.format(output_path))
	detection_model.save_weights('{}/detection_weights.h5'.format(output_path))


	#------------------------------
	# VALIDATION
	#------------------------------

	if num_of_validation_examples > 0:

		print('{}: VALIDATION'.format(datetime.now().time()), end='\n')

		loss = np.zeros((num_of_validation_examples, 1))

		for batch_idx in range(num_of_validation_examples):

			# generate x
			x, img_id = next(gen)

			# get labels
			bbox2d, tmasks = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

			# predict rois
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

			clzbbe_2dtensor = gendy(
				num_of_classes=len(classes), 
				roi_2dtensor=roi_2dtensor, 
				bbox2d=bbox2d, 
				unified_roi_size=unified_roi_size,
				isize=ishape[:2], 
				iou_thresholds=iou_thresholds)

			batch_x = [tf.expand_dims(input=x, axis=0), tf.expand_dims(input=roi_2dtensor, axis=0)]
			batch_y = tf.expand_dims(input=clzbbe_2dtensor, axis=0)

			batch_loss = detection_model.test_on_batch(batch_x, batch_y)
			loss[batch_idx, :] = batch_loss

			print('-', end='')
			if batch_idx%100==99:
				print('{}%'.format(round(batch_idx*100/num_of_validation_examples, 2)))

		print()
		print(np.mean(loss, axis=0))

		with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
			log.write('{}: validation loss: {}\n'.format(datetime.now().time(), np.mean(loss, axis=0)))





