import numpy as np
import tensorflow as tf

from datetime import datetime
from pycocotools.coco import COCO
from datagen import genx, gety
from utils import genanchors, comiou
from models import build_inference_maskrcnn_non_fpn
from settings import settings


params = settings('non-fpn-inference') 

start_example_index 		= params['start_example_index']
num_of_train_examples 		= params['num_of_train_examples']
num_of_validation_examples 	= params['num_of_validation_examples']
num_of_test_examples 		= params['num_of_test_examples']
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
classes 					= params['classes']
mapping 					= params['mapping']
frame_mode 					= params['frame_mode']
ann_file					= params['dataset_anno_file_path']
img_dir						= params['dataset_image_dir_path']
output_path					= params['output_path']

coco = COCO(ann_file)

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
anchor_4dtensor = tf.constant(value=abox4d, dtype='float32')

rpn_model, _ = build_inference_maskrcnn_non_fpn(
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
	base_block_trainable=False)

print('Loads weights')
rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path), by_name=True)

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
		start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
	ishape=ishape)

precision = np.zeros(num_of_test_examples)
false_pred = []

for batch_idx in range(num_of_test_examples):

	x, img_id = next(gen)
	batch_x = tf.expand_dims(input=x, axis=0)

	y_pred = rpn_model.predict_on_batch(batch_x)
	roi2d = np.array(y_pred[0]) # (num_of_rois, 4)
	rois = [list(roi1d) for roi1d in roi2d if roi1d[0] != 0 and roi1d[1] != 0 and roi1d[2] != ishape[0] and roi1d[3] != ishape[1]]
	num_of_rois = len(rois)

	# Get y
	bbox2d, _ = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping) # (num_of_boxes, 4)
	num_of_boxes = bbox2d.shape[0]
	bboxes = [list(bbox[:4]) for bbox in bbox2d]

	true_positives = 0

	for i in range(num_of_rois):
		for j in range(num_of_boxes):
			iou = comiou(bbox=bboxes[j], roi=rois[i])
			true_positives += int(iou >= iou_thresholds[1])

	false_positives = abs(num_of_boxes - true_positives) + abs(num_of_rois - true_positives)
	precision[batch_idx] = true_positives / (true_positives + false_positives)

	if precision[batch_idx] < 1:
		false_pred.append(img_id)

	print('-', end='')
	if batch_idx%100==99:
		print('{}%'.format(round(batch_idx*100/num_of_test_examples, 2)))

print('Precision: {}'.format(np.mean(precision, axis=-1)))

for img_id in false_pred:
	print(img_id, end=',')

















