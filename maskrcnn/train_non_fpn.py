import numpy as np
from datetime import datetime
import tensorflow as tf

from pycocotools.coco import COCO
from models import build_train_maskrcnn_non_fpn
from utils import genanchors, nsm
from datagen import genx, gety, genpy, gendy
from settings import settings


params = settings('non-fpn-train')

start_example_index 		= params['start_example_index']
num_of_train_examples 		= params['num_of_train_examples']
num_of_validation_examples 	= params['num_of_validation_examples']
asizes 						= params['asizes']
ishape 						= params['ishape']
ssize 						= params['ssize']
max_num_of_rois 			= params['max_num_of_rois']
unified_roi_size 			= params['unified_roi_size']
rpn_head_dim 				= params['rpn_head_dim']
fc_denses 					= params['fc_denses']
block_settings 				= params['resnet']
iou_thresholds 				= params['iou_thresholds']
nsm_iou_threshold			= params['nsm_iou_threshold']
nsm_score_threshold			= params['nsm_score_threshold']
num_of_samples				= params['num_of_samples']
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
	base_block_trainable=base_block_trainable)

with open('{}/{}'.format(output_path, log_file_name), 'w') as log:
    detection_model.summary(print_fn=lambda x: log.write(x + '\n'))

# load weihts
if weight_loading is True:
	print('Loads weights')
	rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path))
	detection_model.load_weights('{}/detection_weights.h5'.format(output_path))

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

	loss_rpn = np.zeros((num_of_train_examples, 1))
	loss_det = np.zeros((num_of_train_examples, 1))

	for batch_idx in range(num_of_train_examples):

		# generate x
		x, img_id = next(gen)
		batch_x = tf.expand_dims(input=x, axis=0)

		# get labels
		bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)
		
		# generate y
		clzbbe_4dtensor = genpy(anchor_4dtensor=anchor_4dtensor, bbox2d=bbox2d, iou_thresholds=iou_thresholds, num_of_samples=num_of_samples)
		batch_y = tf.expand_dims(input=clzbbe_4dtensor, axis=0)
			
		batch_loss = rpn_model.train_on_batch(batch_x, batch_y)
		loss_rpn[batch_idx, :] = batch_loss

		# predict rois
		y_pred = rpn_model.predict_on_batch(batch_x)
		clzbbe_4dtensor = y_pred # (batch_size, h, w, 6k), batch_size = 1
		clzbbe_3dtensor = clzbbe_4dtensor[0] # (h, w, 6k)

		roi_2dtensor = nsm(
			anchor_4dtensor=anchor_4dtensor,
			clzbbe_3dtensor=clzbbe_3dtensor,
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
		loss_det[batch_idx, :] = batch_loss

		print('-', end='')
		if batch_idx%100==99:
			print('{}%'.format(round(batch_idx*100/num_of_train_examples, 2)))

	print()
	print(np.mean(loss_rpn, axis=0))
	print(np.mean(loss_det, axis=0))

	with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}: epoch: {}, rpn loss: {}\n{}\n'.format(datetime.now().time(), epoch, np.mean(loss_rpn, axis=0), loss_rpn))
		log.write('{}: epoch: {}, det loss: {}\n{}\n'.format(datetime.now().time(), epoch, np.mean(loss_det, axis=0), loss_det))

	rpn_model.save_weights('{}/rpn_weights.h5'.format(output_path))
	detection_model.save_weights('{}/detection_weights.h5'.format(output_path))


	#------------------------------
	# VALIDATION
	#------------------------------

	if num_of_validation_examples > 0:

		print('{}: VALIDATION'.format(datetime.now().time()), end='\n')

		loss_rpn = np.zeros((num_of_validation_examples, 1))
		loss_det = np.zeros((num_of_validation_examples, 1))

		for batch_idx in range(num_of_validation_examples):

			# generate x
			x, img_id = next(gen)
			batch_x = tf.expand_dims(input=x, axis=0)

			# get labels
			bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)
			
			# generate y
			clzbbe_4dtensor = genpy(anchor_4dtensor=anchor_4dtensor, bbox2d=bbox2d, iou_thresholds=iou_thresholds, num_of_samples=num_of_samples)
			batch_y = tf.expand_dims(input=clzbbe_4dtensor, axis=0)
				
			batch_loss = rpn_model.test_on_batch(batch_x, batch_y)
			loss_rpn[batch_idx, :] = batch_loss

			# predict rois
			y_pred = rpn_model.predict_on_batch(batch_x)
			clzbbe_4dtensor = y_pred # (batch_size, h, w, 6k), batch_size = 1
			clzbbe_3dtensor = clzbbe_4dtensor[0] # (h, w, 6k)

			roi_2dtensor = nsm(
				anchor_4dtensor=anchor_4dtensor,
				clzbbe_3dtensor=clzbbe_3dtensor,
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
			loss_det[batch_idx, :] = batch_loss

			print('-', end='')
			if batch_idx%100==99:
				print('{}%'.format(round(batch_idx*100/num_of_validation_examples, 2)))

		print()
		print(np.mean(loss_rpn, axis=0))
		print(np.mean(loss_det, axis=0))

		with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
			log.write('{}: validation rpn loss: {}\n'.format(datetime.now().time(), np.mean(loss_rpn, axis=0)))
			log.write('{}: validation detection loss: {}\n'.format(datetime.now().time(), np.mean(loss_det, axis=0)))





