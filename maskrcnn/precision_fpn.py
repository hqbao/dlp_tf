import numpy as np
import tensorflow as tf

from datetime import datetime
from pycocotools.coco import COCO
from datagen import genx, gety
from utils import genanchors, comiou
from models import build_inference_maskrcnn_fpn
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
mapping 					= params['mapping']
frame_mode 					= params['frame_mode']
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

rpn_model, detection_model = build_inference_maskrcnn_fpn(
	ishape=ishape, 
	anchor_4dtensors=anchor_4dtensors, 
	classes=classes, 
	max_num_of_rois=max_num_of_rois, 
	nsm_iou_threshold=nsm_iou_threshold, 
	nsm_score_threshold=nsm_score_threshold, 
	unified_roi_size=unified_roi_size,
	fc_denses=fc_denses,
	k0=k0,
	top_down_pyramid_size=top_down_pyramid_size,
	rpn_head_dim=rpn_head_dim,
	block_settings=block_settings,
	base_block_trainable=False)

print('Loads weights')
rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path))
detection_model.load_weights('{}/detection_weights.h5'.format(output_path), by_name=True)

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
		start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
	ishape=ishape)

precision = np.zeros(num_of_test_examples)

for batch_idx in range(num_of_test_examples):

	x, img_id = next(gen)
	batch_x = tf.expand_dims(input=x, axis=0)

	y_pred = detection_model.predict_on_batch(batch_x)
	roi2d = np.array(y_pred[:, 4:]) # (num_of_rois, 5)
	num_of_rois = roi2d.shape[0]

	# Get y
	bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping) # (num_of_boxes, 5)
	num_of_boxes = bbox2d.shape[0]

	true_positives = 0

	for i in range(num_of_rois):
		for j in range(num_of_boxes):
			boxclz1d = bbox2d[j]
			roiclz1d = roi2d[i]
			iou = comiou(bbox=boxclz1d[:4], roi=roiclz1d[:4])
			if iou >= iou_thresholds[1] and boxclz1d[4] == roiclz1d[4]:
				true_positives += 1

	false_positives = abs(num_of_boxes - true_positives)
	precision[batch_idx] = true_positives / (true_positives + false_positives)

	print('-', end='')
	if batch_idx%100==99:
		print('{}%'.format(round(batch_idx*100/num_of_test_examples, 2)))

print('Precision: {}'.format(np.mean(precision, axis=-1)))










