import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from datagen import genx
from utils import genanchors, box2frame, nsm
from models import build_train_maskrcnn_fpn, build_inference_maskrcnn_fpn
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

if False:

	rpn_model, _ = build_train_maskrcnn_fpn(
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

	gen = genx(
		coco=coco, 
		img_dir=img_dir, 
		classes=classes, 
		limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
			start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
		ishape=ishape)

	for _ in range(num_of_test_examples):

		x, _ = next(gen)
		batch_x = tf.expand_dims(input=x, axis=0)
		y_pred = rpn_model.predict_on_batch(batch_x)

		if True:

			_, ax = plt.subplots(2, 2, figsize=(15, 7.35))

			for lvl in range(4):

				anchor_4dtensor = anchor_4dtensors[lvl]
				clzbbe_3dtensor = y_pred[lvl][0] # (h, w, 6k)

				roi_2dtensor = nsm(
					anchor_4dtensor=anchor_4dtensor,
					clzbbe_3dtensor=clzbbe_3dtensor,
					max_num_of_rois=1000,
					nsm_iou_threshold=1.0,
					nsm_score_threshold=nsm_score_threshold,
					ishape=ishape)

				ax[int(lvl/2), lvl%2].imshow(x/255)

				for i in range(roi_2dtensor.shape[0]):
					box = np.array(roi_2dtensor[i], dtype='float32')
					frame = box2frame(box=box, apoint=[0, 0])
					
					ax[int(lvl/2), lvl%2].add_patch(Rectangle(
						(frame[0], frame[1]), frame[2], frame[3],
						linewidth=1, 
						edgecolor='cyan',
						facecolor='none', 
						linestyle='-'))

			plt.show()

		if True:

			_, ax = plt.subplots(2, 2, figsize=(15, 7.35))

			for lvl in range(4):

				anchor_4dtensor = anchor_4dtensors[lvl]
				clzbbe_3dtensor = y_pred[lvl][0] # (h, w, 6k)

				roi_2dtensor = nsm(
					anchor_4dtensor=anchor_4dtensor,
					clzbbe_3dtensor=clzbbe_3dtensor,
					max_num_of_rois=max_num_of_rois,
					nsm_iou_threshold=nsm_iou_threshold,
					nsm_score_threshold=nsm_score_threshold,
					ishape=ishape)

				ax[int(lvl/2), lvl%2].imshow(x/255)

				for i in range(roi_2dtensor.shape[0]):
					box = np.array(roi_2dtensor[i], dtype='float32')
					frame = box2frame(box=box, apoint=[0, 0])
					
					ax[int(lvl/2), lvl%2].add_patch(Rectangle(
						(frame[0], frame[1]), frame[2], frame[3],
						linewidth=1, 
						edgecolor='cyan',
						facecolor='none', 
						linestyle='-'))

			plt.show()

else:

	rpn_model, _ = build_inference_maskrcnn_fpn(
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

	gen = genx(
		coco=coco, 
		img_dir=img_dir, 
		classes=classes, 
		limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
			start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
		ishape=ishape)

	for sample_order in range(num_of_test_examples):

		x, img_id = next(gen)
		batch_x = tf.expand_dims(input=x, axis=0)
		y_pred = rpn_model.predict_on_batch(batch_x)

		_, ax = plt.subplots(figsize=(15, 7.35))

		roi_2dtensor = y_pred[0] # (num_of_rois, 4)

		ax.imshow(x/255)
		ax.set_xlabel('Order: {}, Image ID: {}'.format(sample_order, img_id))
		
		for i in range(roi_2dtensor.shape[0]):
			box = np.array(roi_2dtensor[i], dtype='float32')
			frame = box2frame(box=box, apoint=[0, 0])
			
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=1, 
				edgecolor='cyan',
				facecolor='none', 
				linestyle='-'))

		plt.show()








