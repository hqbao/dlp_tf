import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from utils import normbox2d

class RoiAlignLayer(Layer):
	'''
	'''
	
	def __init__(self, ishape, pool_size, num_of_rois, **kwargs):
		self.ishape = ishape
		self.pool_size = pool_size
		self.num_of_rois = num_of_rois

		super(RoiAlignLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		'''
		Arguments
			input_shape: [feature_map_shape, roi_input_shape]
				feature_map_shape has shape (batch_size, h, w, channels)
				roi_input_shape has shape (batch_size, num_of_rois, 4), 4 is length of [x, y, h, w]
		'''

		assert len(input_shape) == 2, 'RoiAlignLayer must be passed 2 inputs: feature_map_4dtensor and roi_3dtensor'

		self.feature_map_shape = input_shape[0]
		self.rois_input_shape = input_shape[1]

		super(RoiAlignLayer, self).build(input_shape)

	def compute_output_shape(self, input_shape):
		'''
		Arguments
			input_shape: [feature_map_shape, roi_input_shape]
				feature_map_shape has shape (batch_size, h, w, channels)
				roi_input_shape has shape (batch_size, num_of_rois, 4), 4 is length of [x, y, h, w]

		Return
			None, num_of_rois, pool w, pool h, channels
		'''

		assert len(input_shape) == 2, 'RoiAlignLayer must be passed 2 inputs: feature_map_4dtensor and roi_3dtensor'
		
		return None, self.num_of_rois, self.pool_size[0], self.pool_size[1], self.feature_map_shape[3]
		
	def call(self, x):
		'''
		Roi Align https://arxiv.org/pdf/1703.06870.pdf
		Arguments:
			x:
		Return
			resized_roi_5dtensor:
		'''

		assert len(x) == 2, 'RoiAlignLayer must be passed 2 inputs: feature_map_4dtensor and roi_3dtensor'

		feature_map_4dtensor = x[0] # (batch_size, h, w, channels)
		roi_2dtensor = x[1][0] # (num_of_rois, 4)

		crop_size = self.pool_size
		ishape = self.ishape

		roi_indices = tf.zeros(shape=(roi_2dtensor.shape[0]), dtype='int32')
		norm_roi_2dtensor = normbox2d(box_2dtensor=roi_2dtensor, isize=ishape[:2])

		# Stop gradient propogation to ROIs
		norm_roi_2dtensor = tf.stop_gradient(norm_roi_2dtensor)
		roi_indices = tf.stop_gradient(roi_indices)

		resized_roi_4dtensor = tf.image.crop_and_resize(
			image=feature_map_4dtensor,
			boxes=norm_roi_2dtensor,
			box_indices=roi_indices,
			crop_size=tf.constant(value=crop_size),
			method="bilinear") # (num_of_rois, crop_size_height, crop_size_width, channels)

		resized_roi_5dtensor = tf.expand_dims(input=resized_roi_4dtensor, axis=0) # (batch_size, num_of_rois, h, w, channels)

		return resized_roi_5dtensor







