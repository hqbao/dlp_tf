import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, TimeDistributed, Flatten, Dense, Add, BatchNormalization, Activation, UpSampling2D
from RoiAlignLayer import RoiAlignLayer
from PyramidRoiAlignLayer import PyramidRoiAlignLayer
from NSMLayer import NSMLayer
from PyramidNSMLayer import PyramidNSMLayer
from OutputLayer import OutputLayer


def identity_block(input_tensor, kernel_size, filters, trainable=True):
	'''
	https://arxiv.org/pdf/1512.03385.pdf
	Bottleneck architecture
	Arguments
		input_tensor:
		kernel_size:
		filters:
		trainable:
	Return
		tensor:
	'''

	filters1, filters2, filters3 = filters

	tensor = Conv2D(filters=filters1, kernel_size=(1, 1), trainable=trainable)(input_tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', trainable=trainable)(tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(filters=filters3, kernel_size=(1, 1), trainable=trainable)(tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)
	tensor = Add()([tensor, input_tensor])
	tensor = Activation('relu')(tensor)

	return tensor

def conv_block(input_tensor, kernel_size, filters, strides=(1, 1), trainable=True):
	'''
	https://arxiv.org/pdf/1512.03385.pdf
	Bottleneck architecture
	Arguments
		input_tensor:
		kernel_size:
		filters:
		strides:
		trainable:
	Return
		tensor:
	'''

	filters1, filters2, filters3 = filters

	tensor = Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, trainable=trainable)(input_tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', trainable=trainable)(tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(filters=filters3, kernel_size=(1, 1), trainable=trainable)(tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)

	input_tensor = Conv2D(filters=filters3, kernel_size=(1, 1), strides=strides, trainable=trainable)(input_tensor)
	input_tensor = BatchNormalization(trainable=trainable)(input_tensor)

	tensor = Add()([tensor, input_tensor])
	tensor = Activation('relu')(tensor)

	return tensor

def resnet(input_tensor, block_settings, trainable=True):
	'''
	https://arxiv.org/pdf/1512.03385.pdf
	Bottleneck architecture
	Arguments
		input_tensor:
		block_settings:
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]] # Resnet 50, pool 64
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [23, [2, 2]], [3, [2, 2]]] # Resnet 101, pool 64
			[[64, 64, 256], [3, [2, 2]], [8, [2, 2]], [36, [2, 2]], [3, [2, 2]]] # Resnet 152, pool 64
		trainable:
	Return
		tensor:
	'''

	filters, [n_C2, strides_C2], [n_C3, strides_C3], [n_C4, strides_C4], [n_C5, strides_C5] = block_settings
	filters = np.array(filters) # [64, 64, 256] by paper

	tensor = Conv2D(filters=filters[0], kernel_size=(7, 7), strides=(2, 2), padding='same', trainable=trainable)(input_tensor)
	tensor = BatchNormalization(trainable=trainable)(tensor)
	tensor = Activation('relu')(tensor)
	C1 = tensor = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(tensor)

	tensor = conv_block(input_tensor=tensor, kernel_size=(3, 3), filters=filters, strides=strides_C2, trainable=trainable)
	for n in range(n_C2-1):
		tensor = identity_block(input_tensor=tensor, kernel_size=(3, 3), filters=filters, trainable=trainable)
	C2 = tensor

	tensor = conv_block(input_tensor=tensor, kernel_size=(3, 3), filters=2*filters, strides=strides_C3, trainable=trainable)
	for n in range(n_C3-1):
		tensor = identity_block(input_tensor=tensor, kernel_size=(3, 3), filters=2*filters, trainable=trainable)
	C3 = tensor

	tensor = conv_block(input_tensor=tensor, kernel_size=(3, 3), filters=4*filters, strides=strides_C4, trainable=trainable)
	for n in range(n_C4-1):
		tensor = identity_block(input_tensor=tensor, kernel_size=(3, 3), filters=4*filters, trainable=trainable)
	C4 = tensor
	
	tensor = conv_block(input_tensor=tensor, kernel_size=(3, 3), filters=8*filters, strides=strides_C5, trainable=trainable)
	for n in range(n_C5-1):
		tensor = identity_block(input_tensor=tensor, kernel_size=(3, 3), filters=8*filters, trainable=trainable)
	C5 = tensor

	return C1, C2, C3, C4, C5

def non_fpn(input_tensor, block_settings, trainable=True):
	'''
	Non FPN backbone
	Arguments
		input_tensor:
		block_settings:
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]] # Resnet 50, pool 64
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [23, [2, 2]], [3, [2, 2]]] # Resnet 101, pool 64
			[[64, 64, 256], [3, [2, 2]], [8, [2, 2]], [36, [2, 2]], [3, [2, 2]]] # Resnet 152, pool 64
		trainable:
	Return
		tensor:
	'''

	_, _, _, _, C5 = resnet(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		trainable=trainable)
	return C5

def first_backbone(input_tensor, block_settings, trainable=True):
	'''
	Bottom up network in Pyramid Feature Net https://arxiv.org/pdf/1612.03144.pdf
	First backbone in CBNet https://arxiv.org/pdf/1909.03625v1.pdf
	Arguments
		input_tensor:
		block_settings:
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]] # Resnet 50, pool 64
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [23, [2, 2]], [3, [2, 2]]] # Resnet 101, pool 64
			[[64, 64, 256], [3, [2, 2]], [8, [2, 2]], [36, [2, 2]], [3, [2, 2]]] # Resnet 152, pool 64
		trainable:
	Return
		C1:
		C2:
		C3;
		C4;
		C5:
	'''

	C1, C2, C3, C4, C5 = resnet(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		trainable=trainable)
	return C1, C2, C3, C4, C5

def fpn_top_down(C2, C3, C4, C5, top_down_pyramid_size, trainable=True):
	'''
	Top down network in Pyramid Feature Net https://arxiv.org/pdf/1612.03144.pdf
	Arguments
		C2:
		C3:
		C4:
		C5:
		top_down_pyramid_size:
		trainable:
	Return
		P2:
		P3;
		P4;
		P5:
	'''

	M5 = Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(C5)
	M4 = Add()([
		UpSampling2D(size=(2, 2))(M5),
		Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(C4)
	])
	M3 = Add()([
		UpSampling2D(size=(2, 2))(M4),
		Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(C3)
	])
	M2 = Add()([
		UpSampling2D(size=(2, 2))(M3),
		Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(C2)
	])

	P5 = M5
	P4 = Conv2D(filters=top_down_pyramid_size, kernel_size=(3, 3), padding='same', trainable=trainable)(M4)
	P3 = Conv2D(filters=top_down_pyramid_size, kernel_size=(3, 3), padding='same', trainable=trainable)(M3)
	P2 = Conv2D(filters=top_down_pyramid_size, kernel_size=(3, 3), padding='same', trainable=trainable)(M2)

	return P2, P3, P4, P5

def fpn_top_down_balanced(C2, C3, C4, C5, top_down_pyramid_size, trainable=True):
	'''
	Balanced top down network in  Libra R-CNN https://arxiv.org/pdf/1904.02701.pdf
	Arguments
		C2:
		C3:
		C4:
		C5:
		top_down_pyramid_size:
		trainable:
	Return
		P2:
		P3;
		P4;
		P5:
	'''

	D5 = UpSampling2D(size=(8, 8))(C5)
	D4 = UpSampling2D(size=(4, 4))(C4)
	D3 = UpSampling2D(size=(2, 2))(C3)
	D2 = C2
	
	E5 = Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(D5)
	E4 = Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(D4)
	E3 = Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(D3)
	E2 = Conv2D(filters=top_down_pyramid_size, kernel_size=(1, 1), trainable=trainable)(D2)
	
	F = tf.concat(values=[E2, E3, E4, E5], axis=0) # (4, h, w, channels)
	G = tf.math.reduce_mean(input_tensor=F, axis=0, keepdims=True) # (1, h, w, channels)

	H5 = AveragePooling2D(pool_size=(8, 8), trainable=trainable)(G)
	H4 = AveragePooling2D(pool_size=(4, 4), trainable=trainable)(G)
	H3 = AveragePooling2D(pool_size=(2, 2), trainable=trainable)(G)
	H2 = G

	P5 = Add()([C5, H5])
	P4 = Add()([C4, H4])
	P3 = Add()([C3, H3])
	P2 = Add()([C2, H2])

	return P2, P3, P4, P5

def fpn(input_tensor, block_settings, top_down_pyramid_size, trainable=True):
	'''
	Pyramid Feature Net https://arxiv.org/pdf/1612.03144.pdf
	with improvements in Libra R-CNN https://arxiv.org/pdf/1904.02701.pdf
	Arguments
		input_tensor:
		block_settings:
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]] # Resnet 50, pool 64
			[[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [23, [2, 2]], [3, [2, 2]]] # Resnet 101, pool 64
			[[64, 64, 256], [3, [2, 2]], [8, [2, 2]], [36, [2, 2]], [3, [2, 2]]] # Resnet 152, pool 64
		top_down_pyramid_size:
		trainable:
	Return
		P2:
		P3;
		P4;
		P5:
	'''

	_, C2, C3, C4, C5 = first_backbone(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		trainable=trainable)
	P2, P3, P4, P5 = fpn_top_down(
		C2=C2, 
		C3=C3, 
		C4=C4, 
		C5=C5, 
		top_down_pyramid_size=top_down_pyramid_size, 
		trainable=trainable)
	P2, P3, P4, P5 = fpn_top_down_balanced(
		C2=P2, 
		C3=P3, 
		C4=P4, 
		C5=P5, 
		top_down_pyramid_size=top_down_pyramid_size, 
		trainable=trainable)

	return P2, P3, P4, P5

def rpn(input_tensor, k, f):
	'''
	RPN in Faster R-CNN https://arxiv.org/pdf/1506.01497.pdf
	Arguments
		input_tensor:
		k: number of anchors per location on imtermediate Conv2D layer
		f: filters of imtermediate Conv2D layer
	Return
		tensor
	'''

	tensor = Conv2D(filters=f, kernel_size=(3, 3), padding='same')(input_tensor)
	tensor = BatchNormalization()(tensor)
	tensor = Activation('relu')(tensor)
	clz_tensor = Conv2D(filters=2*k,  kernel_size=(1, 1), padding='same', activation='softmax')(tensor) # (batch_size, h, w, 2k)
	bbe_tensor = Conv2D(filters=4*k,  kernel_size=(1, 1), padding='same')(tensor) # (batch_size, h, w, 4k)
	tensor = tf.concat(values=[clz_tensor, bbe_tensor], axis=3) # (batch_size, h, w, 6k)

	return tensor

def nsm_block_non_fpn(input_tensor, ishape, num_of_rois, nsm_iou_threshold, 
	nsm_score_threshold, anchor_4dtensor):
	'''
	Non max suppression in case of non FPN
	Arguments
		input_tensor:
		ishape:
		num_of_rois:
		nsm_iou_threshold:
		nsm_score_threshold:
		anchor_4dtensor:
	Return
		tensor
	'''

	tensor = NSMLayer(
		ishape=ishape, 
		num_of_rois=num_of_rois, 
		nsm_iou_threshold=nsm_iou_threshold, 
		nsm_score_threshold=nsm_score_threshold, 
		anchor_4dtensor=anchor_4dtensor)(input_tensor)
		
	return tensor

def nsm_block_fpn(input_tensors, ishape, num_of_rois, nsm_iou_threshold, 
	nsm_score_threshold, anchor_4dtensors):
	'''
	Non max suppression in case of FPN
	Arguments
		input_tensor:
		ishape:
		num_of_rois:
		nsm_iou_threshold:
		nsm_score_threshold:
		anchor_4dtensor:
	Return
		tensor
	'''

	tensor = PyramidNSMLayer(
		ishape=ishape, 
		num_of_rois=num_of_rois, 
		nsm_iou_threshold=nsm_iou_threshold, 
		nsm_score_threshold=nsm_score_threshold, 
		anchor_4dtensors=anchor_4dtensors)(input_tensors)
		
	return tensor

def classifier_net_non_fpn(input_tensor, ishape, roi_tensor, unified_roi_size, 
	num_of_classes, fc_denses):
	'''
	Classifier net in case of non FPN
	Arguments
		input_tensor:
		ishape:
		roi_tensor:
		unified_roi_size:
		num_of_classes:
		fc_denses:
	Return
		tensor
	'''

	tensor = RoiAlignLayer(
		ishape=ishape, 
		pool_size=unified_roi_size, 
		num_of_rois=roi_tensor.shape[1])([input_tensor, roi_tensor])
	tensor = TimeDistributed(Flatten())(tensor)
	for fc_dense in fc_denses:
		tensor = TimeDistributed(Dense(fc_dense))(tensor)
		tensor = TimeDistributed(BatchNormalization())(tensor)
		tensor = TimeDistributed(Activation('relu'))(tensor)

	clz_tensor = TimeDistributed(Dense(num_of_classes, activation='softmax'))(tensor)
	bbe_tensor = TimeDistributed(Dense(4))(tensor)
	tensor = tf.concat(values=[clz_tensor, bbe_tensor], axis=2) # (batch_size, num_of_rois, num_of_classes+4)

	return tensor

def classifier_net_fpn(input_tensors, ishape, roi_tensor, unified_roi_size, k0, 
	num_of_classes, fc_denses):
	'''
	Classifier net in case of FPN
	Arguments
		input_tensor:
		ishape:
		roi_tensor:
		unified_roi_size:
		num_of_classes:
		fc_denses:
	Return
		tensor
	'''

	tensor = PyramidRoiAlignLayer(
		ishape=ishape, 
		pool_size=unified_roi_size, 
		num_of_rois=roi_tensor.shape[1], k0=k0)([input_tensors, roi_tensor])
	tensor = TimeDistributed(Flatten())(tensor)
	for fc_dense in fc_denses:
		tensor = TimeDistributed(Dense(fc_dense))(tensor)
		tensor = TimeDistributed(BatchNormalization())(tensor)
		tensor = TimeDistributed(Activation('relu'))(tensor)
		
	clz_tensor = TimeDistributed(Dense(num_of_classes, activation='softmax'))(tensor)
	bbe_tensor = TimeDistributed(Dense(4))(tensor)
	tensor = tf.concat(values=[clz_tensor, bbe_tensor], axis=2) # (batch_size, num_of_rois, num_of_classes+4)

	return tensor

def output_block(input_tensor, roi_tensor, num_of_rois, ishape):
	'''
	To compute final output from classification branch and location branch
	Arguments
		input_tensor:
		roi_tensor:
		num_of_rois:
		ishape:
	Return
		tensor
	'''

	tensor = OutputLayer(
		ishape=ishape,
		num_of_rois=num_of_rois)([input_tensor, roi_tensor])
		
	return tensor




