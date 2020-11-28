import tensorflow as tf
import numpy as np
from models import build_robust_test_model
from datagen import genanchors


print('tensorflow version: {}'.format(tf.__version__))

output_path = 'output'
ishape = [512, 512, 3]
ssize = [128, 128]
asizes = [[32, 32]]
total_classes = 1
resnet_settings = [[8, 8, 32], [24, [2, 2]]]
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 100

a1box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes), dtype='float32') # (h1*w1*k1, 4)
a2box_2dtensor = tf.constant(value=genanchors(isize=[ishape[0]//2, ishape[1]//2], ssize=[ssize[0]//2, ssize[1]//2], asizes=asizes), dtype='float32') # (h2*w2*k2, 4)
a3box_2dtensor = tf.constant(value=genanchors(isize=[ishape[0]//4, ishape[1]//4], ssize=[ssize[0]//4, ssize[1]//4], asizes=asizes), dtype='float32') # (h3*w3*k3, 4)
a4box_2dtensor = tf.constant(value=genanchors(isize=[ishape[0]//8, ishape[1]//8], ssize=[ssize[0]//8, ssize[1]//8], asizes=asizes), dtype='float32') # (h4*w4*k4, 4)
abox_2dtensor = tf.concat(values=[a1box_2dtensor, a2box_2dtensor, a3box_2dtensor, a4box_2dtensor], axis=0)

model = build_robust_test_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)

model.load_weights('{}/xsmall/weights_best_precision_recall.h5'.format(output_path), by_name=True)
model.load_weights('{}/small/weights_best_precision_recall.h5'.format(output_path), by_name=True)
model.load_weights('{}/medium/weights_best_precision_recall.h5'.format(output_path), by_name=True)
model.load_weights('{}/large/weights_best_precision_recall.h5'.format(output_path), by_name=True)
model.save('{}/model'.format(output_path))

# Then run this command under output folder
# > tensorflowjs_converter --input_format=tf_saved_model model/ tfjs/