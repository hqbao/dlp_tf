import tensorflow as tf
import numpy as np
from models import build_infer_model
from utils import genanchors


print('tensorflow version: {}'.format(tf.__version__))

output_path = 'output'
ishape = [1024, 1024, 3]
ssizes = [
	[256, 256],
	[128, 128],
	[64, 64], 
	[32, 32], 
]
asizes = [
	[[64, 64]],
	[[128, 128]],
	[[256, 256]],
	[[512, 512]],
]
total_classes = 1
resnet_settings = [[8, 8, 32], [8, [2, 2]], [8, [2, 2]], [8, [2, 2]], [4, [2, 2]]]
top_down_pyramid_size = 64
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 100

a1box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0]), dtype='float32') # (h1*w1*k1, 4)
a2box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1]), dtype='float32') # (h2*w2*k2, 4)
a3box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2]), dtype='float32') # (h3*w3*k3, 4)
a4box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[3], asizes=asizes[3]), dtype='float32') # (h4*w4*k4, 4)
abox_2dtensors = [a1box_2dtensor, a2box_2dtensor, a3box_2dtensor, a4box_2dtensor]
abox_2dtensor = tf.concat(values=abox_2dtensors, axis=0)

model = build_infer_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	top_down_pyramid_size=top_down_pyramid_size,
	k=[len(asizes[0]), len(asizes[1]), len(asizes[2]), len(asizes[3])], 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_recall.h5'.format(output_path), by_name=True)
model.save('{}/model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/model'.format(output_path))
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
open('{}/det_model.tflite'.format(output_path), 'wb').write(tflite_model)




