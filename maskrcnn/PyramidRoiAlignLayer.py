import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from utils import normbox2d

class PyramidRoiAlignLayer(Layer):
	'''
	'''
	
	def __init__(self, ishape, pool_size, num_of_rois, k0, **kwargs):
		self.ishape = ishape
		self.pool_size = pool_size
		self.num_of_rois = num_of_rois
		self.k0 = k0

		super(PyramidRoiAlignLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		'''
		Arguments
			input_shape: [4 feature_map_shapes, roi_input_shape]
				feature_map_shape has shape (batch_size, h, w, channels)
				roi_input_shape has shape (batch_size, num_of_rois, 4), 4 is length of [x, y, h, w]
		'''

		assert len(input_shape) == 2, 'PyramidRoiAlignLayer must be passed 2 inputs: feature_map_4dtensors and roi_3dtensor'

		self.feature_map_shape = input_shape[0]
		self.rois_input_shape = input_shape[1]

		super(PyramidRoiAlignLayer, self).build(input_shape)

	def compute_output_shape(self, input_shape):
		'''
		Arguments
			input_shape: [4 feature_map_shapes, roi_input_shape]
				feature_map_shape has shape (batch_size, h, w, channels)
				roi_input_shape has shape (batch_size, num_of_rois, 4), 4 is length of [x, y, h, w]

		Return
			None, num_of_rois, pool w, pool h, channels
		'''

		assert len(input_shape) == 2, 'PyramidRoiAlignLayer must be passed 2 inputs: feature_map_4dtensors and roi_3dtensor'
		
		return None, self.num_of_rois, self.pool_size[0], self.pool_size[1], self.feature_map_shape[3]
		
	def call(self, x):
		'''
		Roi Align https://arxiv.org/pdf/1703.06870.pdf
		Arguments:
			x:
		Return
			resized_roi_5dtensor:
		'''

		assert len(x) == 2, 'PyramidRoiAlignLayer must be passed 2 inputs: feature_map_4dtensors and roi_3dtensor'

		feature_map_4dtensors = x[0] # (4, batch_size, h, w, channels), batch_size = 1
		roi_2dtensor = x[1][0] # (num_of_rois, 4), batch_size = 1

		k0 = self.k0 # min_anchor_size_squared = 2^k0, e.g. 8 = 2^3
		crop_size = self.pool_size
		ishape = self.ishape

		y1, x1, y2, x2 = tf.split(value=roi_2dtensor, num_or_size_splits=4, axis=1) # (num_of_rois, 1)
		h = y2 - y1
		w = x2 - x1
		roi_level_2dtensor = tf.math.log(tf.sqrt(h*w))/tf.math.log(2.0)
		roi_level_2dtensor = tf.math.minimum(3, tf.math.maximum(0, tf.cast(tf.math.round(roi_level_2dtensor - k0), dtype='int32'))) # (num_of_rois, 1)
		roi_level_2dtensor = tf.squeeze(input=roi_level_2dtensor, axis=1) # (num_of_rois,)
		norm_roi_2dtensor = normbox2d(box_2dtensor=roi_2dtensor, isize=ishape[:2])

		resized_roi_4dtensors = []

		for lvl in range(4):
			feature_map_4dtensor = feature_map_4dtensors[lvl]
			roi_indices = tf.where(condition=tf.math.equal(x=roi_level_2dtensor, y=lvl))
			lvl_roi_2dtensor = tf.gather_nd(params=norm_roi_2dtensor, indices=roi_indices)
			roi_indices *= 0 # batch_size = 1, indices should be all zeros
			roi_indices = tf.cast(x=roi_indices[:, 0], dtype='int32')

			# Stop gradient propogation to ROIs
			lvl_roi_2dtensor = tf.stop_gradient(lvl_roi_2dtensor)
			roi_indices = tf.stop_gradient(roi_indices)

			resized_roi_4dtensor = tf.image.crop_and_resize(
				image=feature_map_4dtensor,
				boxes=lvl_roi_2dtensor,
				box_indices=roi_indices,
				crop_size=tf.constant(value=crop_size),
				method="bilinear")

			resized_roi_4dtensors.append(resized_roi_4dtensor)

		resized_roi_4dtensor = tf.concat(values=resized_roi_4dtensors, axis=0) # (num_of_rois, crop_size_height, crop_size_width, channels)
		resized_roi_5dtensor = tf.expand_dims(input=resized_roi_4dtensor, axis=0) # (batch_size, num_of_rois, h, w, channels), batch_size = 1

		return resized_roi_5dtensor







