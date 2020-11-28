import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Add, TimeDistributed, UpSampling2D, GaussianNoise
from tensorflow.keras.backend import categorical_crossentropy, switch
from utils import nsm


def identity_block(input_tensor, kernel_size, filters, block_name, use_bias, weight_decay, trainable, bn_trainable):
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

	tensor = Conv2D(
		filters=filters1, 
		kernel_size=[1, 1], 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv1')(input_tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv1_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(
		filters=filters2, 
		kernel_size=kernel_size, 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv2')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv2_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(
		filters=filters3, 
		kernel_size=[1, 1], 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv3')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv3_bn')(tensor)
	tensor = Add()([tensor, input_tensor])
	tensor = Activation('relu')(tensor)

	return tensor

def conv_block(input_tensor, kernel_size, filters, strides, block_name, use_bias, weight_decay, trainable, bn_trainable):
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

	tensor = Conv2D(
		filters=filters1, 
		kernel_size=[1, 1], 
		strides=strides, 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv1')(input_tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv1_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(
		filters=filters2, 
		kernel_size=kernel_size, 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv2')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv2_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(
		filters=filters3, 
		kernel_size=[1, 1], 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv3')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv3_bn')(tensor)

	input_tensor = Conv2D(
		filters=filters3, 
		kernel_size=[1, 1], 
		strides=strides, 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv4')(input_tensor)
	input_tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'_conv4_bn')(input_tensor, training=trainable)

	tensor = Add()([tensor, input_tensor])
	tensor = Activation('relu')(tensor)

	return tensor

def resnet(input_tensor, block_settings, use_bias, weight_decay, trainable, bn_trainable):
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

	filters = np.array(block_settings[0])
	n_C2, strides_C2 = block_settings[1]
	tensors = []

	# C1
	tensor = Conv2D(
		filters=filters[0], 
		kernel_size=[7, 7], 
		strides=[2, 2], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay),
		trainable=trainable, 
		name='conv1')(input_tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name='conv1_bn')(tensor)
	tensor = Activation('relu')(tensor)

	# C2
	tensor = MaxPool2D(pool_size=[3, 3], strides=strides_C2, padding='same')(tensor)
	tensor = conv_block(
		input_tensor=tensor, 
		kernel_size=[3, 3], 
		filters=filters, 
		strides=[1, 1], 
		block_name='stg1_blk0_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)
	for n in range(1, n_C2):
		tensor = identity_block(
			input_tensor=tensor, 
			kernel_size=[3, 3], 
			filters=filters, 
			block_name='stg1_blk'+str(n)+'_', 
			use_bias=use_bias, 
			weight_decay=weight_decay,
			trainable=trainable,
			bn_trainable=bn_trainable)

	tensors.append(tensor)

	# C34...
	for c in range(2, 2+len(block_settings[2:])):
		n_C, strides_C = block_settings[c]
		tensor = conv_block(
			input_tensor=tensor, 
			kernel_size=[3, 3], 
			filters=(2**(c-1))*filters, 
			strides=strides_C, 
			block_name='stg'+str(c)+'_blk0_', 
			use_bias=use_bias, 
			weight_decay=weight_decay,
			trainable=trainable,
			bn_trainable=bn_trainable)
		for n in range(1, n_C):
			tensor = identity_block(
				input_tensor=tensor, 
				kernel_size=[3, 3], 
				filters=(2**(c-1))*filters, 
				block_name='stg'+str(c)+'_blk'+str(n)+'_', 
				use_bias=use_bias, 
				weight_decay=weight_decay,
				trainable=trainable,
				bn_trainable=bn_trainable)

		tensors.append(tensor)

	return tensors

def RFE(input_tensor, module_name, use_bias, weight_decay, trainable, bn_trainable):
	'''
	'''

	tensors = []
	kernel_sizes = [[1, 3], [1, 5], [3, 1], [5, 1]]
	top_down_pyramid_size = input_tensor.shape[-1]

	for i in range(len(kernel_sizes)):
		tensor = Conv2D(
			filters=top_down_pyramid_size//4, 
			kernel_size=[1, 1], 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=module_name+'_conv1_'+str(i))(input_tensor)
		tensor = BatchNormalization(trainable=bn_trainable, name=module_name+'_conv1_'+str(i)+'_bn')(tensor)
		tensor = Activation('relu')(tensor)

		tensor = Conv2D(
			filters=top_down_pyramid_size//4, 
			kernel_size=kernel_sizes[i], 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=module_name+'_conv2_'+str(i))(tensor)
		tensor = BatchNormalization(trainable=bn_trainable, name=module_name+'_conv2_'+str(i)+'_bn')(tensor)
		tensor = Activation('relu')(tensor)

		tensor = Conv2D(
			filters=top_down_pyramid_size//4, 
			kernel_size=[1, 1], 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=module_name+'_conv3_'+str(i))(tensor)
		tensor = BatchNormalization(trainable=bn_trainable, name=module_name+'_conv3_'+str(i)+'_bn')(tensor)
		tensor = Activation('relu')(tensor)

		tensors.append(tensor)

	tensor = tf.concat(values=tensors, axis=-1)
	tensor = Conv2D(
			filters=top_down_pyramid_size, 
			kernel_size=[1, 1], 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=module_name+'rfe_conv4')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=module_name+'_conv4_bn')(tensor)
	tensor = Activation('relu')(tensor)
	tensor = Add()([tensor, input_tensor])

	return tensor

def fpn_top_down(input_tensors, top_down_pyramid_size, use_bias, weight_decay, trainable, bn_trainable):
	'''
	Top down network in Pyramid Feature Net https://arxiv.org/pdf/1612.03144.pdf
	Arguments
		input_tensors:
		top_down_pyramid_size:
		trainable:
	Return
		P2:
		P3;
		P4;
	'''

	C2, C3, C4 = input_tensors

	L3 = Conv2D(
		filters=top_down_pyramid_size, 
		kernel_size=[3, 3], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='lateral_P3')(C3)
	L3 = BatchNormalization(trainable=bn_trainable, name='lateral_P3_bn')(L3)
	L3 = Activation('relu')(L3)
	L2 = Conv2D(
		filters=top_down_pyramid_size, 
		kernel_size=[3, 3], 
		padding='same', 
		trainable=trainable, 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		name='lateral_P2')(C2)
	L2 = BatchNormalization(trainable=bn_trainable, name='lateral_P2_bn')(L2)
	L2 = Activation('relu')(L2)

	M4 = Conv2D(
		filters=top_down_pyramid_size, 
		kernel_size=[3, 3], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='M4')(C4)
	M4 = BatchNormalization(trainable=bn_trainable, name='M4_bn')(M4)
	M4 = Activation('relu')(M4)
	M3 = Add(name='M3')([UpSampling2D(size=(2, 2), interpolation='bilinear')(M4), L3])
	M2 = Add(name='M2')([UpSampling2D(size=(2, 2), interpolation='bilinear')(M3), L2])

	P4 = RFE(input_tensor=M4, module_name='RFE1', trainable=trainable, bn_trainable=bn_trainable, use_bias=use_bias, weight_decay=weight_decay)
	P3 = RFE(input_tensor=M3, module_name='RFE2', trainable=trainable, bn_trainable=bn_trainable, use_bias=use_bias, weight_decay=weight_decay)
	P2 = RFE(input_tensor=M2, module_name='RFE3', trainable=trainable, bn_trainable=bn_trainable, use_bias=use_bias, weight_decay=weight_decay)

	return [P2, P3, P4]

def fpn(input_tensor, resnet_settings, top_down_pyramid_size, use_bias, weight_decay, trainable, bn_trainable):
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
		bn_trainable:
	Return
		P2:
		P3;
		P4;
		P5:
	'''

	tensors = resnet(
		input_tensor=input_tensor, 
		block_settings=resnet_settings, 
		use_bias=use_bias, 
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)

	P2, P3, P4 = fpn_top_down(
		input_tensors=tensors, 
		top_down_pyramid_size=top_down_pyramid_size, 
		use_bias=use_bias, 
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)

	return [P2, P3, P4]

def loss(total_classes, lamda=1.0):
	'''
	'''

	def smooth_l1(y_true, y_pred):
		'''
		'''

		HUBER_DELTA = 1.0

		x = tf.math.abs(y_true - y_pred)
		x = switch(x < HUBER_DELTA, 0.5*x**2, HUBER_DELTA*(x - 0.5*HUBER_DELTA))
		return  x

	def ssd_loss(y_true, y_pred):
		'''
		https://arxiv.org/pdf/1512.02325.pdf
		Arguments
			y_true: (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1+4)
			y_pred: (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1+4)
		Return
			loss
		'''

		true_clz_2dtensor = y_true[:, :total_classes+1] # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1)
		pred_clz_2dtensor = y_pred[:, :total_classes+1] # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1)
		true_loc_2dtensor = y_true[:, total_classes+1:] # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, 4)
		pred_loc_2dtensor = y_pred[:, total_classes+1:] # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, 4)

		sum_true_clz_2dtensor = tf.math.reduce_sum(input_tensor=true_clz_2dtensor, axis=-1) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3,)
		selected_clz_indices = tf.where(
			condition=tf.math.equal(x=sum_true_clz_2dtensor, y=1)) # foreground, background
		selected_loc_indices = tf.where(
			condition=tf.math.logical_and(
				x=tf.math.equal(x=sum_true_clz_2dtensor, y=1),
				y=tf.math.not_equal(x=true_clz_2dtensor[:, -1], y=1))) # foreground

		true_clz_2dtensor = tf.gather_nd(params=true_clz_2dtensor, indices=selected_clz_indices) # (fb, total_classes+1)
		pred_clz_2dtensor = tf.gather_nd(params=pred_clz_2dtensor, indices=selected_clz_indices) # (fb, total_classes+1)
		true_loc_2dtensor = tf.gather_nd(params=true_loc_2dtensor, indices=selected_loc_indices) # (f, 4)
		pred_loc_2dtensor = tf.gather_nd(params=pred_loc_2dtensor, indices=selected_loc_indices) # (f, 4)

		clz_loss = categorical_crossentropy(true_clz_2dtensor, pred_clz_2dtensor) # (fb,)
		loc_loss = tf.math.reduce_sum(input_tensor=smooth_l1(true_loc_2dtensor, pred_loc_2dtensor), axis=-1) # (f,)
		loss = tf.math.reduce_mean(clz_loss) + lamda*tf.math.reduce_mean(loc_loss)

		return loss

	return ssd_loss

def build_model(ishape, resnet_settings, top_down_pyramid_size, k, total_classes):
	'''
	'''

	use_bias = True
	weight_decay = 0.0
	trainable = True
	bn_trainable = True

	input_tensor = Input(shape=ishape, name='input', dtype='float32')

	nets = fpn(
		input_tensor=input_tensor, 
		resnet_settings=resnet_settings, 
		top_down_pyramid_size=top_down_pyramid_size, 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable, 
		bn_trainable=bn_trainable)

	tensors = []
	for i in range(len(nets)):
		tensor = nets[i]

		tensor = Conv2D(
			filters=k[i]*(total_classes+1+4), 
			kernel_size=[1, 1], 
			strides=[1, 1], 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name='head'+str(i)+'_conv')(tensor)
		tensor = tf.reshape(tensor=tensor, shape=[-1, total_classes+1+4]) # (h*w*k, total_classes+1+4)
		clz_tensor = tensor[:, :total_classes+1]
		clz_tensor = Activation('softmax')(clz_tensor)
		loc_tensor = tensor[:, total_classes+1:]
		tensor = tf.concat(values=[clz_tensor, loc_tensor], axis=-1) # (h*w*k, total_classes+1+4)
		tensors.append(tensor)

	tensor = tf.concat(values=tensors, axis=0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1+4)

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(), loss=loss(total_classes=total_classes, lamda=1.0))

	return model

def build_infer_model(ishape, resnet_settings, top_down_pyramid_size, k, total_classes, abox_2dtensor, nsm_iou_threshold, nsm_score_threshold, nsm_max_output_size):
	'''
	'''

	use_bias = True
	weight_decay = 0.0
	trainable = False
	bn_trainable = False

	input_tensor = Input(shape=ishape, name='input', dtype='float32')
	
	nets = fpn(
		input_tensor=input_tensor, 
		resnet_settings=resnet_settings, 
		top_down_pyramid_size=top_down_pyramid_size, 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable, 
		bn_trainable=bn_trainable)

	tensors = []
	for i in range(len(nets)):
		tensor = nets[i]
		
		tensor = Conv2D(
			filters=k[i]*(total_classes+1+4), 
			kernel_size=[1, 1], 
			strides=[1, 1], 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name='head'+str(i)+'_conv')(tensor)
		tensor = tf.reshape(tensor=tensor, shape=[tensor.shape[1]*tensor.shape[2]*k[i], total_classes+1+4]) # (h*w*k, total_classes+1+4)
		clz_tensor = tensor[:, :total_classes+1]
		clz_tensor = Activation('softmax')(clz_tensor)
		loc_tensor = tensor[:, total_classes+1:]
		tensor = tf.concat(values=[clz_tensor, loc_tensor], axis=-1) # (h*w*k, total_classes+1+4)
		tensors.append(tensor)

	tensor = tf.concat(values=tensors, axis=0) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1+4)

	tensor, valid_outputs = nsm(
		abox_2dtensor=abox_2dtensor, 
		prediction=tensor, 
		nsm_iou_threshold=nsm_iou_threshold,
		nsm_score_threshold=nsm_score_threshold,
		nsm_max_output_size=nsm_max_output_size,
		total_classes=total_classes)
	valid_outputs = tf.expand_dims(input=valid_outputs, axis=0)

	model = Model(inputs=input_tensor, outputs=[tensor, valid_outputs])
	model.compile(optimizer=Adam(), loss=None)

	return model







