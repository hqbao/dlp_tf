import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from models import build_inference_maskrcnn_non_fpn
from utils import genanchors, box2frame
from settings import settings


params = settings('non-fpn-inference')

asizes 						= params['asizes']
ishape 						= params['ishape']
ssize 						= params['ssize']
max_num_of_rois 			= params['max_num_of_rois']
unified_roi_size 			= params['unified_roi_size']
rpn_head_dim				= params['rpn_head_dim']
fc_denses 					= params['fc_denses']
block_settings 				= params['resnet']
nsm_iou_threshold			= params['nsm_iou_threshold']
nsm_score_threshold			= params['nsm_score_threshold']
classes 					= params['classes']
output_path					= params['output_path']

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

rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path), by_name=True)
# detection_model.load_weights('{}/detection_weights.h5'.format(output_path), by_name=True)

rpn_model.save('{}/rpn_model'.format(output_path))
# detection_model.save('{}/detection_model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/rpn_model'.format(output_path))
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('{}/rpn_model.tflite'.format(output_path), 'wb').write(tflite_model)

# converter = tf.lite.TFLiteConverter.from_saved_model('{}/detection_model'.format(output_path))
# converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open('{}/detection_model.tflite'.format(output_path), 'wb').write(tflite_model)








