import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from models import build_inference_maskrcnn_non_fpn
from utils import genanchors, box2frame
from datagen import genx
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
ann_file					= params['dataset_anno_file_path']
img_dir						= params['dataset_image_dir_path']
output_path					= params['output_path']

coco = COCO(ann_file)

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
anchor_4dtensor = tf.constant(value=abox4d, dtype='float32')

rpn_model, detection_model = build_inference_maskrcnn_non_fpn(
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
detection_model.load_weights('{}/detection_weights.h5'.format(output_path), by_name=True)

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=(start_example_index+num_of_train_examples+num_of_validation_examples, 
		start_example_index+num_of_train_examples+num_of_validation_examples+num_of_test_examples),
	ishape=ishape)

for sample_order in range(num_of_test_examples):

	print('{}: START'.format(datetime.now().time()), end='\n')

	x, img_id = next(gen)

	# predict object proposal
	batch_x = tf.expand_dims(input=x, axis=0)
	y_pred = detection_model.predict_on_batch(batch_x)

	print('{}: END'.format(datetime.now().time()), end='\n')

	_, ax = plt.subplots(figsize=(15, 7.35))

	num_of_rois = y_pred.shape[0]

	ax.imshow(x/255)
	ax.set_xlabel('Order: {}, Image ID: {}'.format(sample_order, img_id))
	
	for k in range(num_of_rois):
		pclz = int(y_pred[k, 8])
		
		# if pclz == len(classes)-1:
		# 	continue

		rbox = np.array(y_pred[k, :4])
		pbox = np.array(y_pred[k, 4:8])

		tframe = box2frame(box=pbox, apoint=[0.5, 0.5])
		
		ax.annotate(classes[pclz], (tframe[0], tframe[1]), color='green', weight='bold',
			fontsize=8, ha='center', va='center')

		rframe = box2frame(box=rbox, apoint=[0, 0])
		tframe = box2frame(box=pbox, apoint=[0, 0])

		ax.add_patch(Rectangle(
			(rframe[0], rframe[1]), rframe[2], rframe[3],
			linewidth=1, 
			edgecolor='r',
			facecolor='none', 
			linestyle='-'))

		ax.add_patch(Rectangle(
			(tframe[0], tframe[1]), tframe[2], tframe[3],
			linewidth=1, 
			edgecolor='g',
			facecolor='none', 
			linestyle='-'))

	plt.show()











