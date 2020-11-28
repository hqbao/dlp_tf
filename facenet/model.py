import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Add, UpSampling2D


def identity_block(input_tensor, kernel_size, filters, block_name, use_bias=True, weight_decay=0.0001, trainable=True, bn_trainable=True):
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

def conv_block(input_tensor, kernel_size, filters, strides, block_name, use_bias=True, weight_decay=0.0001, trainable=True, bn_trainable=True):
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

def resnet(input_tensor, block_settings, use_bias=True, weight_decay=0.0001, trainable=True, bn_trainable=True):
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

	# C345
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

	return tensor

def embedding_net(input_tensor, resnet_settings, dense_settings, use_bias=True, weight_decay=0.0001, trainable=True, bn_trainable=True):
	'''
	'''

	tensor = resnet(
		input_tensor=input_tensor, 
		block_settings=resnet_settings, 
		use_bias=use_bias, 
		weight_decay=weight_decay, 
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor = AveragePooling2D(pool_size=[7, 7], strides=[1, 1], padding='valid')(tensor)
	tensor = Flatten()(tensor)

	for i in range(len(dense_settings)):
		dense_units, dropout_rate = dense_settings[i]
		tensor = Dense(
			units=dense_units, 
			use_bias=use_bias, 
			trainable=trainable, 
			name='fc'+str(i))(tensor)
		tensor = Activation('relu')(tensor)
		tensor = Dropout(rate=dropout_rate)(tensor, training=trainable)

	return tensor

def build_clz_model(ishape, resnet_settings, dense_settings, total_classes):
	'''
	'''

	learning_rate = 0.001
	use_bias = True
	weight_decay = 0.0001
	trainable = True
	bn_trainable = True

	input_tensor = Input(shape=ishape, name='input')
	tensor = embedding_net(
		input_tensor=input_tensor, 
		resnet_settings=resnet_settings, 
		dense_settings=dense_settings,
		use_bias=use_bias,
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = Dense(
		units=total_classes, 
		use_bias=use_bias, 
		trainable=trainable, 
		name='fc_identity', 
		kernel_regularizer=regularizers.l2(weight_decay))(tensor)
	tensor = Activation('softmax')(tensor)

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=tf.keras.losses.categorical_crossentropy)

	return model

def triplet_loss(alpha=0.2):
	'''
	'''

	def loss(y_true, y_pred):
		'''
		http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
		https://arxiv.org/pdf/1503.03832v3.pdf
		Arguments
			y_true: (3*batch_size, embedding_dims)
			y_pred: (3*batch_size, embedding_dims)
		Return
			loss
		'''

		fa_2dtensor = y_pred[0::3, :] # (batch_size, embedding_dims)
		fp_2dtensor = y_pred[1::3, :] # (batch_size, embedding_dims)
		fn_2dtensor = y_pred[2::3, :] # (batch_size, embedding_dims)
		diff_pos_2dtensor = fa_2dtensor - fp_2dtensor # (batch_size, embedding_dims)
		diff_neg_2dtensor = fa_2dtensor - fn_2dtensor # (batch_size, embedding_dims)
		squared_diff_pos_2dtensor = tf.math.square(x=diff_pos_2dtensor) # (batch_size, embedding_dims)
		squared_diff_neg_2dtensor = tf.math.square(x=diff_neg_2dtensor) # (batch_size, embedding_dims)
		squared_pos_dist = tf.math.reduce_sum(input_tensor=squared_diff_pos_2dtensor, axis=-1) # (batch_size,)
		squared_neg_dist = tf.math.reduce_sum(input_tensor=squared_diff_neg_2dtensor, axis=-1) # (batch_size,)
		loss = tf.math.maximum(x=squared_pos_dist + alpha - squared_neg_dist, y=0.0) # (batch_size,)
		loss = tf.math.reduce_mean(input_tensor=loss, axis=-1)

		return loss

	return loss

def build_ver_model(ishape, resnet_settings, dense_settings, embedding_dims, alpha):
	'''
	'''
	
	learning_rate = 0.001
	use_bias = True
	weight_decay = 0.0001
	trainable = False
	bn_trainable = False

	input_tensor = Input(shape=ishape, name='input')
	tensor = embedding_net(
		input_tensor=input_tensor, 
		resnet_settings=resnet_settings, 
		dense_settings=dense_settings,
		use_bias=use_bias,
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)

	tensor = tf.math.l2_normalize(x=tensor, axis=-1)
	tensor = Dense(
		units=embedding_dims, 
		use_bias=False, 
		name='fc_learned_embedding', 
		kernel_regularizer=regularizers.l2(weight_decay))(tensor)

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=triplet_loss(alpha=alpha))

	return model

def build_rec_model(ishape, resnet_settings, dense_settings):
	'''
	'''
	
	
	learning_rate = 0.001
	use_bias = True
	weight_decay = 0.0001
	trainable = False
	bn_trainable = False

	input_tensor = Input(shape=ishape, name='input')
	tensor = embedding_net(
		input_tensor=input_tensor, 
		resnet_settings=resnet_settings, 
		dense_settings=dense_settings,
		use_bias=use_bias,
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor = tf.math.l2_normalize(x=tensor, axis=-1)

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: 0.0)

	return model

def build_idgen_model(ishape, resnet_settings, dense_settings):
	'''
	'''
	
	learning_rate = 0.001
	use_bias = True
	weight_decay = 0.0001
	trainable = False
	bn_trainable = False

	input_tensor = Input(shape=ishape, name='input', dtype='int32')
	tensor = tf.cast(x=input_tensor, dtype='float32')
	tensor = embedding_net(
		input_tensor=tensor, 
		resnet_settings=resnet_settings, 
		dense_settings=dense_settings,
		use_bias=use_bias,
		weight_decay=weight_decay,
		trainable=trainable,
		bn_trainable=bn_trainable)
	tensor = tf.math.l2_normalize(x=tensor, axis=-1) # (batch_size, embedding_dims)

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=lambda y_true, y_pred: 0.0)

	return model

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
		y_true: (batch_size, h, w, 5)
		y_pred: (batch_size, h, w, 5)
	Return
		loss
	'''

	_, total_heatmaps, h, w = y_pred.shape

	y_true = tf.reshape(tensor=y_true, shape=[-1, total_heatmaps*h*w])
	y_pred = tf.reshape(tensor=y_pred, shape=[-1, total_heatmaps*h*w])

	# cancel_bg = tf.where(condition=tf.math.greater(x=y_true, y=0.1), x=1.0, y=0.0)

	diff = y_true - y_pred # (batch_size, h*w*5)
	diff = tf.math.abs(x=diff) # (batch_size, h*w*5)

	loss = tf.where(
		condition=tf.math.less(x=diff, y=1),
		x=0.5*tf.math.square(x=diff),
		y=diff-0.5) # (batch_size, h*w*5)
	loss = tf.math.reduce_sum(input_tensor=loss, axis=-1) # (batch_size,)
	loss = tf.math.reduce_mean(input_tensor=loss, axis=-1) 

	# fg_loss = diff
	# fg_loss = tf.math.log(1.0+fg_loss)/tf.math.log(2.0) # (batch_size, h*w*5)
	# fg_loss *= cancel_bg # (batch_size, h*w*5)
	# fg_loss = tf.math.reduce_sum(input_tensor=fg_loss, axis=-1) # (batch_size,)
	# fg_loss = tf.math.reduce_mean(input_tensor=fg_loss, axis=-1) 

	return loss

def build_ali_model(ishape, mode='train'):
	'''
	'''
	
	learning_rate = 0.001
	use_bias = True
	weight_decay = 0.0001
	trainable = True if mode == 'train' else False
	bn_trainable = True if mode == 'train' else False

	input_tensor = Input(shape=ishape, name='input', dtype='int32')
	itensor = tf.cast(x=input_tensor, dtype='float32')
	tensor = Conv2D(
		filters=16, 
		kernel_size=[3, 3], 
		strides=[1, 1], 
		padding='same',
		use_bias=use_bias, 
		kernel_regularizer=regularizers.l2(weight_decay), 
		trainable=trainable, 
		name='conv1')(itensor)
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
		filters=5, 
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

	loss = ali_loss
	if mode != 'train':
		heatmap_4dtensor = tensor - itensor/255
		tensor = heatmap_4dtensor[0] # (h, w, 5), batch_size = 1
		tensor = tf.where(condition=tf.math.greater(x=tensor, y=0.8), x=tensor, y=tensor*0)
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

	model = Model(inputs=input_tensor, outputs=tensor)
	model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

	return model







