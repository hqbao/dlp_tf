import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from utils import bbe2box2d

class OutputLayer(Layer):
	'''
	'''
	
	def __init__(self, ishape, num_of_rois, **kwargs):
		self.ishape = ishape
		self.num_of_rois = num_of_rois
		super(OutputLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		'''
		Arguments
			input_shape: [clzbbe_tensor, roi_tensor]
		'''

		assert len(input_shape) == 2, 'OutputLayer must be passed 2 inputs: clzbbe_tensor and roi_tensor'

		super(OutputLayer, self).build(input_shape)

	def compute_output_shape(self, input_shape):
		'''
		Arguments
			input_shape: [clzbbe_tensor, roi_tensor]

		Return
			None, num_of_rois, 9
		'''
		
		return None, self.num_of_rois, 9
		
	def call(self, x):
		'''
		To compute final output from classification branch and location branch
		Arguments
			x:
		Return
			tensor
		'''

		assert len(x) == 2, 'OutputLayer must be passed 2 inputs: clzbbe_tensor and roi_tensor'

		clzbbe_3dtensor = x[0] # (batch_size, num_of_rois, num_of_classes+4), batch_size = 1
		num_of_class = clzbbe_3dtensor.shape[2]-4 
		clz_2dtensor = clzbbe_3dtensor[0, :, :num_of_class] # (num_of_rois, num_of_classes)
		bbe_2dtensor = clzbbe_3dtensor[0, :, num_of_class:] # (num_of_rois, 4)
		roi_3dtensor = x[1] # (batch_size, num_of_rois, 4), batch_size = 1
		roi_2dtensor = roi_3dtensor[0] # (num_of_rois, 4)

		pbox_2dtensor = bbe2box2d(box_2dtensor=roi_2dtensor, bbe_2dtensor=bbe_2dtensor)
		clz_3dtensor = tf.expand_dims(input=clz_2dtensor, axis=2)
		pclzprob = tf.math.reduce_max(input_tensor=clz_3dtensor, axis=1)
		pclz = tf.math.argmax(input=clz_3dtensor, axis=1)
		pclz = tf.cast(pclz, dtype='float32')

		t = tf.concat(values=[roi_2dtensor, pbox_2dtensor, pclz, pclzprob], axis=1)

		return t












