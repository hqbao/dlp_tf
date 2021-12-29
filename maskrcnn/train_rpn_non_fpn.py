import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO
from datagen import genx, gety, genpy
from utils import genanchors
from models import build_train_maskrcnn_non_fpn
from settings import settings
from datetime import datetime


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

log_file_name = "{}:rpn-train-log.txt".format(datetime.now().time())

#------------------------------
# RPN TRAIN & VALIDATION
#------------------------------

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
anchor_4dtensor = tf.constant(value=abox4d, dtype='float32')

rpn_model, _ = build_train_maskrcnn_non_fpn(
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
    rpn_model.summary(print_fn=lambda x: log.write(x + '\n'))


# load weihts
if weight_loading is True:
	print('Loads weights')
	rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path))


for epoch in range(num_of_epoches):

	gen = genx(
		coco=coco,
		img_dir=img_dir,
		classes=classes,
		limit=(start_example_index, 
			start_example_index+num_of_train_examples+num_of_validation_examples),
		ishape=ishape)

	#------------------------------
	# RPN TRAIN
	#------------------------------

	loss = np.zeros((num_of_train_examples, 1))

	print('{}: TRAIN {}'.format(datetime.now().time(), epoch), end='\n')

	for batch_idx in range(num_of_train_examples):

		# generate x
		x, img_id = next(gen)
		batch_x = tf.expand_dims(input=x, axis=0)

		# Get y
		bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

		# generate y
		clzbbe_4dtensor = genpy(anchor_4dtensor=anchor_4dtensor, bbox2d=bbox2d[:, :4], iou_thresholds=iou_thresholds, num_of_samples=num_of_samples)
		batch_y = tf.expand_dims(input=clzbbe_4dtensor, axis=0)
		
		batch_loss = rpn_model.train_on_batch(batch_x, batch_y)
		loss[batch_idx, :] = batch_loss

		print('-', end='')
		if batch_idx%100==99:
			print('{}%'.format(round(batch_idx*100/num_of_train_examples, 2)))

	print()
	print(np.mean(loss, axis=0))

	with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
		log.write('{}: epoch: {}, train loss: {}\n'.format(datetime.now().time(), epoch, np.mean(loss, axis=0)))

	# save weights
	rpn_model.save_weights('{}/rpn_weights.h5'.format(output_path))



	if num_of_validation_examples > 0:

		#------------------------------
		# RPN VALIDATION
		#------------------------------

		loss = np.zeros((num_of_validation_examples, 1))

		print('{}: VALIDATION {}'.format(datetime.now().time(), epoch), end='\n')

		for batch_idx in range(num_of_validation_examples):
			
			# generate x
			x, img_id = next(gen)
			batch_x = tf.expand_dims(input=x, axis=0)

			# Get y
			bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

			# generate y
			clzbbe_4dtensor = genpy(anchor_4dtensor=anchor_4dtensor, bbox2d=bbox2d[:, :4], iou_thresholds=iou_thresholds, num_of_samples=num_of_samples)
			batch_y = tf.expand_dims(input=clzbbe_4dtensor, axis=0)
			
			batch_loss = rpn_model.test_on_batch(batch_x, batch_y)
			loss[batch_idx, :] = batch_loss

			print('-', end='')
			if batch_idx%100==99:
				print('{}%'.format(round(batch_idx*100/num_of_validation_examples, 2)))	

		print()
		print(np.mean(loss, axis=0))

		with open('{}/{}'.format(output_path, log_file_name), 'a') as log:
			log.write('{}: validation loss: {}\n'.format(datetime.now().time(), np.mean(loss, axis=0)))







