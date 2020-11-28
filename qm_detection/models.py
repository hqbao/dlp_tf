import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Add
from tensorflow.keras.backend import categorical_crossentropy, sum, mean, abs, switch
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

def loss(total_classes, lamda=1.0):
	'''
	'''

	def smooth_l1(y_true, y_pred):
		'''
		'''

		HUBER_DELTA = 1.0

		x = abs(y_true - y_pred)
		x = switch(x < HUBER_DELTA, 0.5*x**2, HUBER_DELTA*(x - 0.5*HUBER_DELTA))
		return  x

	def ssd_loss(y_true, y_pred):
		'''
		https://arxiv.org/pdf/1512.02325.pdf
		Arguments
			y_true: (h*w*k, total_classes+1+4)
			y_pred: (h*w*k, total_classes+1+4)
		Return
			loss
		'''

		true_clz_2dtensor = y_true[:, :total_classes+1] # (h*w*k, total_classes+1)
		pred_clz_2dtensor = y_pred[:, :total_classes+1] # (h*w*k, total_classes+1)
		true_loc_2dtensor = y_true[:, total_classes+1:] # (h*w*k, 4)
		pred_loc_2dtensor = y_pred[:, total_classes+1:] # (h*w*k, 4)

		sum_true_clz_2dtensor = tf.math.reduce_sum(input_tensor=true_clz_2dtensor, axis=-1) # (h*w*k,)
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
		loc_loss = sum(smooth_l1(true_loc_2dtensor, pred_loc_2dtensor), axis=-1) # (f,)
		loss = mean(clz_loss) + lamda*mean(loc_loss)

		return loss

	return ssd_loss

def build_model(ishape, resnet_settings, k, total_classes):
	'''
	'''

	use_bias = True
	weight_decay = 0.0
	trainable = True
	bn_trainable = True

	input_tensor = Input(shape=ishape, name='input', dtype='float32')

	tensors = resnet(
		input_tensor=input_tensor, 
		block_settings=resnet_settings, 
		use_bias=use_bias, 
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor = tensors[-1]

	head_dims = tensor.shape[3]
	tensor = Conv2D(
		filters=head_dims, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='conv6')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name='conv6_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(
		filters=k*(total_classes+1+4), 
		kernel_size=[1, 1], 
		strides=[1, 1], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='conv7')(tensor)
	tensor = tf.reshape(tensor=tensor, shape=[-1, total_classes+1+4]) # (h*w*k, total_classes+1+4)
	clz_tensor = tensor[:, :total_classes+1]
	clz_tensor = Activation('softmax')(clz_tensor)
	loc_tensor = tensor[:, total_classes+1:]
	output_tensor = tf.concat(values=[clz_tensor, loc_tensor], axis=-1) # (h*w*k, total_classes+1+4)

	model = Model(inputs=input_tensor, outputs=output_tensor)
	model.compile(optimizer=Adam(), loss=loss(total_classes=total_classes, lamda=1.0))

	return model

def build_infer_model(ishape, resnet_settings, k, total_classes, abox_2dtensor, nsm_iou_threshold, nsm_score_threshold, nsm_max_output_size):
	'''
	'''

	use_bias = True
	weight_decay = 0.0
	trainable = False
	bn_trainable = False

	input_tensor = Input(shape=ishape, name='input', dtype='uint8')
	tensor = tf.cast(x=input_tensor, dtype='float32')

	tensors = resnet(
		input_tensor=tensor, 
		block_settings=resnet_settings, 
		use_bias=use_bias, 
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor = tensors[-1]

	head_dims = tensor.shape[3]
	tensor = Conv2D(
		filters=head_dims, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='conv6')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name='conv6_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = Conv2D(
		filters=k*(total_classes+1+4), 
		kernel_size=[1, 1], 
		strides=[1, 1], 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable,
		name='conv7')(tensor)
	tensor = tf.reshape(tensor=tensor, shape=[-1, total_classes+5]) # (h*w*k, total_classes+1+4)
	clz_tensor = tensor[:, :1+total_classes]
	clz_tensor = Activation('softmax')(clz_tensor)
	loc_tensor = tensor[:, 1+total_classes:]
	tensor = tf.concat(values=[clz_tensor, loc_tensor], axis=-1) # (h*w*k, total_classes+1+4)
	
	tensor, valid_outputs = nsm(
		abox_2dtensor=abox_2dtensor, 
		prediction=tensor, 
		nsm_iou_threshold=nsm_iou_threshold,
		nsm_score_threshold=nsm_score_threshold,
		nsm_max_output_size=nsm_max_output_size,
		total_classes=total_classes)
	valid_outputs = tf.cast(x=valid_outputs, dtype='float32')
	valid_outputs = tf.expand_dims(input=valid_outputs, axis=0)

	model = Model(inputs=input_tensor, outputs=[tensor, valid_outputs])
	model.compile(optimizer=Adam(), loss=[lambda y_true, y_pred: 0.0, lambda y_true, y_pred: 0.0])

	return model






