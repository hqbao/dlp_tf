import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Add, UpSampling2D


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

def ali_loss(y_true, y_pred):
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

	loss = tf.where(
		condition=tf.math.less(x=diff, y=1),
		x=0.5*tf.math.square(x=diff),
		y=diff-0.5) # (batch_size, h*w*total_heatmaps)
	loss = tf.math.reduce_sum(input_tensor=loss, axis=-1) # (batch_size,)
	loss = tf.math.reduce_mean(input_tensor=loss, axis=-1)

	return loss

def build_model(ishape, mode='train'):
	'''
	'''
	
	learning_rate = 0.001
	use_bias = True
	weight_decay = 0.0001
	trainable = True if mode == 'train' else False
	bn_trainable = True if mode == 'train' else False

	input_tensor = Input(shape=ishape, name='input', dtype='int32')
	tensor = tf.cast(x=input_tensor, dtype='float32')

	tensor = Conv2D(
		filters=4, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='conv1')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name='conv1_bn')(tensor)
	tensor = Activation('relu')(tensor)

	tensor = hourglass(
		input_tensor=tensor, 
		block_name='blk1_', 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = Conv2D(
		filters=2, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		activation='relu',
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='conv2')(tensor)
	tensor = BatchNormalization(trainable=bn_trainable, name='conv2_bn')(tensor)
	tensor = Activation('relu')(tensor)

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=ali_loss)

	return model







