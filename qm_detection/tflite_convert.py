import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_infer_model
from datagen import genx
from utils import genanchors
from datetime import datetime


print('tensorflow version: {}'.format(tf.__version__))

ishape = [240, 200, 3]
ssize = [60, 50]
asizes = [[8, 8]]
resnet_settings = [[5, 5, 20], [2, [1, 1]], [8, [2, 2]]]
total_classes = 2
output_path = 'output'
nsm_iou_threshold = 0.1
nsm_score_threshold = 0.9
nsm_max_output_size = 330

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
abox_4dtensor = tf.constant(value=abox4d, dtype='float32')
abox_2dtensor = tf.reshape(tensor=abox_4dtensor, shape=[-1, 4])

model = build_infer_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_recall.h5'.format(output_path), by_name=True)
model.save('{}/model'.format(output_path))

def representative_dataset():
	for _ in range(1):
		yield [np.zeros([240, 200, 3], dtype='uint8')]

converter = tf.lite.TFLiteConverter.from_saved_model('{}/model'.format(output_path))
converter.experimental_new_converter = True
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
tflite_model = converter.convert()
open('{}/model.tflite'.format(output_path), 'wb').write(tflite_model)
