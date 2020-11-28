import tensorflow as tf
import numpy as np
from models import build_model, nsm
from datagen import genanchors, comiou, load_dataset, genxy, genxy_com


output_path = 'output/xsmall'
ishape = [512, 512, 3] # [64, 64, 3], [128, 128, 3], [256, 256, 3], [512, 512, 3]
asizes = [[32, 32]]
total_classes = 1
resnet_settings = [[8, 8, 32], [24, [2, 2]]]

model = build_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	net_name='XNet')
# model.summary()
model.load_weights('{}/weights_best_precision.h5'.format(output_path))
model.save_weights('{}/weights_.h5'.format(output_path))
