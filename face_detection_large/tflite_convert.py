import tensorflow as tf
import numpy as np
from models import build_test_model
from utils import genanchors


print('Tensorflow version: {}'.format(tf.__version__))

output_path = 'output'
ishape = [64, 64, 3]
ssize = [16, 16]
asizes = [[32, 32]]
total_classes = 1
resnet_settings = [[8, 8, 32], [32, [2, 2]]]
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 100

abox_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes), dtype='float32') # (h*w*k, 4)

model = build_test_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_precision.h5'.format(output_path), by_name=True)
model.save('{}/model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/model'.format(output_path))
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
open('{}/det_model.tflite'.format(output_path), 'wb').write(tflite_model)




