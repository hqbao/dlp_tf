import tensorflow as tf
import numpy as np
from models import build_infer_model
from utils import genanchors


print('tensorflow version: {}'.format(tf.__version__))

output_path = 'output'
ishape = [256, 256, 3]
ssizes = [
	[64, 64], 
	[32, 32], 
	[16, 16],
]
asizes = [
	[[32, 32]],
	[[64, 64]],
	[[128, 128]],
]
total_classes = 1
resnet_settings = [[16, 16, 64], [4, [2, 2]], [8, [2, 2]], [16, [2, 2]]]
top_down_pyramid_size = 64
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 10

a1box2d = genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0]) # (h1 * w1 * k1, 4)
a2box2d = genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1]) # (h2 * w2 * k2, 4)
a3box2d = genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2]) # (h3 * w3 * k3, 4)
abox2d = np.concatenate([a1box2d, a2box2d, a3box2d], axis=0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, 4)
abox_2dtensor = tf.constant(value=abox2d, dtype='float32') # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, 4)

model = build_infer_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	top_down_pyramid_size=top_down_pyramid_size,
	k=[len(asizes[0]), len(asizes[1]), len(asizes[2])], 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_precision.h5'.format(output_path), by_name=True)
model.save('{}/model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/model'.format(output_path))
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
# converter.allow_custom_ops=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_types = [tf.uint8]
# converter.inference_input_type = tf.uint8  # or tf.uint8
# converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_model = converter.convert()
open('{}/det_model.tflite'.format(output_path), 'wb').write(tflite_model)




