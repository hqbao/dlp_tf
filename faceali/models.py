import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Add, TimeDistributed, UpSampling2D, GaussianNoise
from tensorflow.keras.backend import categorical_crossentropy, switch, sum, mean


def convstg(input_tensor, total_layers, filters, strides, stage_name, use_bias, weight_decay, trainable, bn_trainable):
	'''
	'''

	tensor = input_tensor
	for i in range(total_layers):
		strides = strides if i == 0 else [1, 1]
		tensor = Conv2D(
			filters=filters, 
			kernel_size=[3, 3], 
			strides=strides, 
			padding='same',
			use_bias=use_bias, 
			kernel_regularizer=regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=stage_name+'_conv'+str(i+1))(tensor)
		tensor = BatchNormalization(trainable=bn_trainable, name=stage_name+'_bn'+str(i+1))(tensor)
		tensor = Activation('relu')(tensor)

	return tensor

def hourglass(input_tensor, block_name, use_bias, weight_decay, trainable, bn_trainable):
	'''
	'''

	filters = input_tensor.shape[3]
	tensor = input_tensor

	tensor = convstg(
		input_tensor=tensor, 
		total_layers=2, 
		filters=filters, 
		strides=[1, 1],
		stage_name=block_name+'stg1_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor1 = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[1, 1],
		stage_name=block_name+'stgx1_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[2, 2],
		stage_name=block_name+'stg2_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor2 = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[1, 1],
		stage_name=block_name+'stgx2_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[2, 2],
		stage_name=block_name+'stg3_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor3 = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[1, 1],
		stage_name=block_name+'stgx3_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[2, 2],
		stage_name=block_name+'stg4_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor4 = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[1, 1],
		stage_name=block_name+'stgx4_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = convstg(
		input_tensor=tensor, 
		total_layers=3, 
		filters=filters, 
		strides=[2, 2],
		stage_name=block_name+'stg5_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = UpSampling2D(size=(2, 2))(tensor)
	tensor = Add()([tensor, tensor4])
	tensor = Conv2D(
		filters=filters, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'stgup4_conv')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'stgup4_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = UpSampling2D(size=(2, 2))(tensor)
	tensor = Add()([tensor, tensor3])
	tensor = Conv2D(
		filters=filters, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'stgup3_conv')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'stgup3_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = UpSampling2D(size=(2, 2))(tensor)
	tensor = Add()([tensor, tensor2])
	tensor = Conv2D(
		filters=filters, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'stgup2_conv')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'stgup2_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = UpSampling2D(size=(2, 2))(tensor)
	tensor = Add()([tensor, tensor1])
	tensor = Conv2D(
		filters=filters, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'stgup1_conv1')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'stgup1_bn1')(tensor)
	tensor = Activation('relu')(tensor)
	tensor = Conv2D(
		filters=filters, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'stgup1_conv2')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=block_name+'stgup1_bn2')(tensor)
	tensor = Activation('relu')(tensor)

	return tensor

def heatmap_loss(y_true, y_pred):
	'''
	Arguments
		y_true: (batch_size, h, w, total_heatmaps)
		y_pred: (batch_size, h, w, total_heatmaps)
	Return
		loss
	'''

	_, h, w, total_heatmaps = y_pred.shape

	y_true = tf.reshape(tensor=y_true, shape=[-1, h*w*total_heatmaps])
	y_pred = tf.reshape(tensor=y_pred, shape=[-1, h*w*total_heatmaps])

	diff = y_true - y_pred # (batch_size, h*w*total_heatmaps)
	diff = tf.math.abs(x=diff) # (batch_size, h*w*total_heatmaps)
	loss = tf.math.reduce_sum(input_tensor=diff, axis=-1) # (batch_size,)
	loss = tf.math.reduce_mean(input_tensor=loss, axis=-1) 

	return loss

def build_model(ishape, mode='train', net_name='Fansipan'):
	'''
	'''
	
	use_bias = True
	weight_decay = 0.0
	trainable = True if mode == 'train' else False
	bn_trainable = True if mode == 'train' else False

	itensor = Input(shape=ishape, name='input', dtype='float32')
	tensor = Conv2D(
		filters=16, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=net_name+'_conv1')(itensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=net_name+'_conv1_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = hourglass(
		input_tensor=tensor, 
		block_name=net_name+'_blk1_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = Conv2D(
		filters=5, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		activation='relu',
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=net_name+'_conv2')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name=net_name+'_conv2_bn')(tensor)
	tensor = Activation('relu')(tensor)

	loss = heatmap_loss
	if mode != 'train':
		heatmap_4dtensor = tensor - itensor
		tensor = heatmap_4dtensor[0] # (h, w, 5), batch_size = 1
		tensor = tf.where(condition=tf.math.greater(x=tensor, y=192), x=tensor, y=tensor*0)
		tensor = tf.transpose(a=tensor, perm=[2, 0, 1]) # (5, h, w)
		hm1 = tf.reshape(tensor=tensor[0], shape=[-1])
		hm2 = tf.reshape(tensor=tensor[1], shape=[-1])
		hm3 = tf.reshape(tensor=tensor[2], shape=[-1])
		hm4 = tf.reshape(tensor=tensor[3], shape=[-1])
		hm5 = tf.reshape(tensor=tensor[4], shape=[-1])
		a = tf.math.argmax(input=hm1)
		b = tf.math.argmax(input=hm2)
		c = tf.math.argmax(input=hm3)
		d = tf.math.argmax(input=hm4)
		e = tf.math.argmax(input=hm5)
		tensor = [a, b, c, d, e]
		loss = None

	model = Model(inputs=itensor, outputs=tensor)
	model.compile(optimizer=Adam(), loss=loss)

	return model

