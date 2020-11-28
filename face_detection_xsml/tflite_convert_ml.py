import tensorflow as tf
import numpy as np
from models import build_ml_test_model
from datagen import genanchors


print('tensorflow version: {}'.format(tf.__version__))

output_path = 'output'
ishape = [128, 128, 3]
ssize = [32, 32]
asizes = [[32, 32]]
total_classes = 1
resnet_settings = [[8, 8, 32], [24, [2, 2]]]
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 10

a1box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes), dtype='float32') # (h1*w1*k1, 4)
a2box_2dtensor = tf.constant(value=genanchors(isize=[ishape[0]//2, ishape[1]//2], ssize=[ssize[0]//2, ssize[1]//2], asizes=asizes), dtype='float32') # (h2*w2*k2, 4)
abox_2dtensor = tf.concat(values=[a1box_2dtensor, a2box_2dtensor], axis=0)

model = build_ml_test_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)

model.summary()
model.load_weights('{}/medium/weights_best_precision_recall.h5'.format(output_path), by_name=True)
model.load_weights('{}/large/weights_best_precision_recall.h5'.format(output_path), by_name=True)
model.save('{}/model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/model'.format(output_path))
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
open('{}/det_ml_model.tflite'.format(output_path), 'wb').write(tflite_model)

