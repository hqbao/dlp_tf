import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean, sum, categorical_crossentropy, abs, switch


def smooth_l1(y_true, y_pred):
	HUBER_DELTA = 1.0

	x = abs(y_true - y_pred)
	x = switch(x < HUBER_DELTA, 0.5*x**2, HUBER_DELTA*(x - 0.5*HUBER_DELTA))
	return  x

def balanced_l1(y_true, y_pred):
	'''
	https://arxiv.org/pdf/1904.02701.pdf
	'''

	alpha = 0.5
	gamma = 1.5
	b = 19.085
	C = 0

	x = abs(y_true - y_pred)
	x = switch(x < 1.0, (alpha*x + alpha/b)*tf.math.log(b*x + 1) - alpha*x, gamma*x + C)
	return  x

def rpn_loss(y_true, y_pred):
	'''
	Arguments
		y_true: (batch_size, h, w, 6k)
		y_pred: (batch_size, h, w, 6k)
	Return
		loss
	'''

	K = y_pred.shape[3]//6

	true_clz_4dtensor = y_true[:, :, :, :2*K] # (batch_size, h, w, 2k)
	true_bbe_4dtensor = y_true[:, :, :, 2*K:] # (batch_size, h, w, 4k)
	pred_clz_4dtensor = y_pred[:, :, :, :2*K] # (batch_size, h, w, 2k)
	pred_bbe_4dtensor = y_pred[:, :, :, 2*K:] # (batch_size, h, w, 4k)

	true_clz_2dtensor = tf.reshape(tensor=true_clz_4dtensor, shape=[-1, 2]) # (h*w*k, 2)
	true_bbe_2dtensor = tf.reshape(tensor=true_bbe_4dtensor, shape=[-1, 4]) # (h*w*k, 4)
	pred_clz_2dtensor = tf.reshape(tensor=pred_clz_4dtensor, shape=[-1, 2]) # (h*w*k, 2)
	pred_bbe_2dtensor = tf.reshape(tensor=pred_bbe_4dtensor, shape=[-1, 4]) # (h*w*k, 4)

	# add small value when output is zeros, avoid log(0) = -inf
	pred_clz_2dtensor = tf.where(
		condition=tf.math.equal(x=pred_clz_2dtensor, y=0.0),
		x=0.00001,
		y=pred_clz_2dtensor)

	LAMBDA = 1.0
	L_clz = categorical_crossentropy(target=true_clz_2dtensor, output=pred_clz_2dtensor) # (h*w*k)
	L_bbe = balanced_l1(true_bbe_2dtensor, pred_bbe_2dtensor) # (h*w*k, 4)
	L_bbe = sum(x=L_bbe, axis=-1) # (h*w*k)
	L = mean(L_clz) + LAMBDA*mean(true_clz_2dtensor[:, 0]*L_bbe)

	return L

def det_loss(y_true, y_pred):
	'''
	Arguments
		y_true: (batch_size, num_of_rois, num_of_classes+4)
		y_pred: (batch_size, num_of_rois, num_of_classes+4)
	Return
		loss
	'''

	num_of_classes = y_pred.shape[2]-4

	true_clz_3dtensor = y_true[:, :, :num_of_classes] # (batch_size, num_of_rois, num_of_classes)
	true_bbe_3dtensor = y_true[:, :, num_of_classes:] # (batch_size, num_of_rois, 4)
	pred_clz_3dtensor = y_pred[:, :, :num_of_classes] # (batch_size, num_of_rois, num_of_classes)
	pred_bbe_3dtensor = y_pred[:, :, num_of_classes:] # (batch_size, num_of_rois, 4)

	true_clz_2dtensor = tf.reshape(tensor=true_clz_3dtensor, shape=[-1, num_of_classes]) # (num_of_rois, num_of_classes)
	true_bbe_2dtensor = tf.reshape(tensor=true_bbe_3dtensor, shape=[-1, 4]) # (num_of_rois, 4)
	pred_clz_2dtensor = tf.reshape(tensor=pred_clz_3dtensor, shape=[-1, num_of_classes]) # (num_of_rois, num_of_classes)
	pred_bbe_2dtensor = tf.reshape(tensor=pred_bbe_3dtensor, shape=[-1, 4]) # (num_of_rois, 4)

	# add small value when output is zeros, avoid log(0) = -inf
	pred_clz_2dtensor = tf.where(
		condition=tf.math.equal(x=pred_clz_2dtensor, y=0.0),
		x=0.0001, 
		y=pred_clz_2dtensor)

	valid_3dtensor = tf.math.reduce_max(input_tensor=true_clz_3dtensor, axis=2, keepdims=True) # (batch_size, num_of_rois, 1)
	valid_1dtensor = tf.reshape(tensor=valid_3dtensor, shape=[-1]) # (num_of_rois)

	LAMBDA = 1.0
	L_clz = categorical_crossentropy(target=true_clz_2dtensor, output=pred_clz_2dtensor) # (num_of_rois)
	L_bbe = balanced_l1(true_bbe_2dtensor, pred_bbe_2dtensor) # (num_of_rois, 4)
	L_bbe = sum(x=L_bbe, axis=-1) # (num_of_rois)
	L = mean(L_clz) + LAMBDA*mean(valid_1dtensor*L_bbe)

	return L

def dumpy_loss(y_true, y_pred):
	'''
	For inference, loss is no need
	'''
	return 0.0

def build_train_maskrcnn_non_fpn(ishape, anchor_4dtensor, classes, max_num_of_rois, 
	nsm_iou_threshold, nsm_score_threshold, unified_roi_size, rpn_head_dim, fc_denses, 
	block_settings, base_block_trainable=True):
	'''
	Arguments
		ishape:
		anchor_4dtensor:
		classes:
		max_num_of_rois:
		nsm_iou_threshold:
		nsm_score_threshold:
		unified_roi_size:
		rpn_head_dim:
		fc_denses:
		block_settings:
		base_block_trainable:
	Return
		rpn_model:
		detection_model:
	'''

	from net_blocks import non_fpn, rpn, classifier_net_non_fpn

	num_of_classes = len(classes)
	k = anchor_4dtensor.shape[2]

	input_tensor = Input(shape=ishape)
	roi_tensor = Input(shape=(max_num_of_rois, 4))

	fmap_tensor = non_fpn(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		trainable=base_block_trainable)

	rpn_clzbbe_tensor = rpn(input_tensor=fmap_tensor, k=k, f=rpn_head_dim)

	clzbbe_tensor = classifier_net_non_fpn(
		input_tensor=fmap_tensor, 
		ishape=ishape, 
		roi_tensor=roi_tensor, 
		unified_roi_size=unified_roi_size, 
		num_of_classes=num_of_classes, 
		fc_denses=fc_denses)

	rpn_model = Model(
		inputs=input_tensor, 
		outputs=rpn_clzbbe_tensor, 
		name='RPN')

	detection_model = Model(
		inputs=[input_tensor, roi_tensor], 
		outputs=clzbbe_tensor, 
		name='DETECTION')

	rpn_model.compile(
		optimizer=Adam(lr=0.001),
		loss=rpn_loss)

	detection_model.compile(
		optimizer=Adam(lr=0.001),
		loss=det_loss)

	return rpn_model, detection_model

def build_inference_maskrcnn_non_fpn(ishape, anchor_4dtensor, classes, max_num_of_rois, 
	nsm_iou_threshold, nsm_score_threshold, unified_roi_size, rpn_head_dim, fc_denses, 
	block_settings, base_block_trainable=False):
	'''
	Arguments
		ishape:
		anchor_4dtensor:
		classes:
		max_num_of_rois:
		nsm_iou_threshold:
		nsm_score_threshold:
		unified_roi_size:
		rpn_head_dim:
		fc_denses:
		block_settings:
		base_block_trainable:
	Return
		rpn_model:
		detection_model:
	'''

	from net_blocks import non_fpn, rpn, nsm_block_non_fpn, classifier_net_non_fpn, output_block

	num_of_classes = len(classes)
	k = anchor_4dtensor.shape[2]

	input_tensor = Input(shape=ishape)
	fmap_tensor = non_fpn(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		trainable=base_block_trainable)

	rpn_clzbbe_tensor = rpn(input_tensor=fmap_tensor, k=k, f=rpn_head_dim)

	roi_tensor = nsm_block_non_fpn(
		input_tensor=rpn_clzbbe_tensor, 
		ishape=ishape,
		num_of_rois=max_num_of_rois, 
		nsm_iou_threshold=nsm_iou_threshold, 
		nsm_score_threshold=nsm_score_threshold, 
		anchor_4dtensor=anchor_4dtensor)

	clzbbe_tensor = classifier_net_non_fpn(
		input_tensor=fmap_tensor, 
		ishape=ishape, 
		roi_tensor=roi_tensor, 
		unified_roi_size=unified_roi_size, 
		num_of_classes=num_of_classes, 
		fc_denses=fc_denses)

	output_tensor = output_block(
		input_tensor=clzbbe_tensor, 
		roi_tensor=roi_tensor, 
		num_of_rois=max_num_of_rois, 
		ishape=ishape)

	rpn_model = Model(inputs=input_tensor, outputs=roi_tensor, name='RPN')
	detection_model = Model(inputs=input_tensor, outputs=output_tensor, name='DETECTION')

	rpn_model.compile(optimizer=Adam(lr=0.001), loss=dumpy_loss)
	detection_model.compile(optimizer=Adam(lr=0.001), loss=dumpy_loss)

	return rpn_model, detection_model

def build_train_maskrcnn_fpn(ishape, anchor_4dtensors, classes, max_num_of_rois, 
	nsm_iou_threshold, nsm_score_threshold, unified_roi_size, k0, top_down_pyramid_size, 
	rpn_head_dim, fc_denses, block_settings, base_block_trainable=True):
	'''
	Arguments
		ishape:
		anchor_4dtensors:
		classes:
		max_num_of_rois:
		nsm_iou_threshold:
		nsm_score_threshold:
		unified_roi_size:
		k0:
		top_down_pyramid_size:
		rpn_head_dim:
		fc_denses:
		block_settings:
		base_block_trainable:
	Return
		rpn_model:
		detection_model:
	'''

	from net_blocks import fpn, rpn, classifier_net_fpn

	num_of_classes = len(classes)
	k1 = anchor_4dtensors[0].shape[2]
	k2 = anchor_4dtensors[0].shape[2]
	k3 = anchor_4dtensors[0].shape[2]
	k4 = anchor_4dtensors[0].shape[2]

	input_tensor = Input(shape=ishape)
	roi_tensor = Input(shape=(max_num_of_rois, 4))
	P2, P3, P4, P5 = fpn(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		top_down_pyramid_size=top_down_pyramid_size, 
		trainable=base_block_trainable)

	lvl1_rpn_clzbbe_tensor = rpn(input_tensor=P2, k=k1, f=rpn_head_dim)
	lvl2_rpn_clzbbe_tensor = rpn(input_tensor=P3, k=k2, f=rpn_head_dim)
	lvl3_rpn_clzbbe_tensor = rpn(input_tensor=P4, k=k3, f=rpn_head_dim)
	lvl4_rpn_clzbbe_tensor = rpn(input_tensor=P5, k=k4, f=rpn_head_dim)

	clzbbe_tensor = classifier_net_fpn(
		input_tensors=[P2, P3, P4, P5], 
		ishape=ishape, 
		roi_tensor=roi_tensor, 
		unified_roi_size=unified_roi_size, 
		k0=k0, 
		num_of_classes=num_of_classes, 
		fc_denses=fc_denses)

	rpn_model = Model(
		inputs=input_tensor, 
		outputs=[
			lvl1_rpn_clzbbe_tensor,
			lvl2_rpn_clzbbe_tensor,
			lvl3_rpn_clzbbe_tensor,
			lvl4_rpn_clzbbe_tensor], 
		name='RPN')

	detection_model = Model(
		inputs=[input_tensor, roi_tensor], 
		outputs=clzbbe_tensor, 
		name='DETECTION')

	rpn_model.compile(
		optimizer=Adam(lr=0.001),
		loss=[
			rpn_loss,
			rpn_loss,
			rpn_loss,
			rpn_loss])

	detection_model.compile(
		optimizer=Adam(lr=0.001),
		loss=det_loss)

	return rpn_model, detection_model

def build_inference_maskrcnn_fpn(ishape, anchor_4dtensors, classes, max_num_of_rois, 
	nsm_iou_threshold, nsm_score_threshold, unified_roi_size, k0, top_down_pyramid_size, 
	rpn_head_dim, fc_denses, block_settings, base_block_trainable=False):
	'''
	Arguments
		ishape:
		anchor_4dtensors:
		classes:
		max_num_of_rois:
		nsm_iou_threshold:
		nsm_score_threshold:
		unified_roi_size:
		k0:
		top_down_pyramid_size:
		rpn_head_dim:
		fc_denses:
		block_settings:
		base_block_trainable:
	Return
		rpn_model:
		detection_model:
	'''

	from net_blocks import fpn, rpn, nsm_block_fpn, classifier_net_fpn, output_block

	num_of_classes = len(classes)
	k1 = anchor_4dtensors[0].shape[2]
	k2 = anchor_4dtensors[1].shape[2]
	k3 = anchor_4dtensors[2].shape[2]
	k4 = anchor_4dtensors[3].shape[2]

	input_tensor = Input(shape=ishape)
	P2, P3, P4, P5 = fpn(
		input_tensor=input_tensor, 
		block_settings=block_settings, 
		top_down_pyramid_size=top_down_pyramid_size, 
		trainable=base_block_trainable)

	lvl1_rpn_clzbbe_tensor = rpn(input_tensor=P2, k=k1, f=rpn_head_dim)
	lvl2_rpn_clzbbe_tensor = rpn(input_tensor=P3, k=k2, f=rpn_head_dim)
	lvl3_rpn_clzbbe_tensor = rpn(input_tensor=P4, k=k3, f=rpn_head_dim)
	lvl4_rpn_clzbbe_tensor = rpn(input_tensor=P5, k=k4, f=rpn_head_dim)

	roi_tensor = nsm_block_fpn(
		input_tensors=[
			lvl1_rpn_clzbbe_tensor,
			lvl2_rpn_clzbbe_tensor,
			lvl3_rpn_clzbbe_tensor,
			lvl4_rpn_clzbbe_tensor], 
		ishape=ishape,
		num_of_rois=max_num_of_rois, 
		nsm_iou_threshold=nsm_iou_threshold, 
		nsm_score_threshold=nsm_score_threshold, 
		anchor_4dtensors=anchor_4dtensors)

	clzbbe_tensor = classifier_net_fpn(
		input_tensors=[P2, P3, P4, P5], 
		ishape=ishape, 
		roi_tensor=roi_tensor, 
		unified_roi_size=unified_roi_size, 
		k0=k0, 
		num_of_classes=num_of_classes, 
		fc_denses=fc_denses)

	output_tensor = output_block(
		input_tensor=clzbbe_tensor, 
		roi_tensor=roi_tensor, 
		num_of_rois=max_num_of_rois, 
		ishape=ishape)

	rpn_model = Model(inputs=input_tensor, outputs=roi_tensor, name='RPN')
	detection_model = Model(inputs=input_tensor, outputs=output_tensor, name='DETECTION')

	rpn_model.compile(optimizer=Adam(lr=0.001), loss=dumpy_loss)
	detection_model.compile(optimizer=Adam(lr=0.001), loss=dumpy_loss)

	return rpn_model, detection_model





	
