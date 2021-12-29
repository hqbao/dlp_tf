import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from utils import pnsm

class PyramidNSMLayer(Layer):
	'''
	'''
	
	def __init__(self, ishape, num_of_rois, nsm_iou_threshold, nsm_score_threshold, anchor_4dtensors, **kwargs):
		self.ishape = ishape
		self.num_of_rois = num_of_rois
		self.nsm_iou_threshold = nsm_iou_threshold
		self.nsm_score_threshold = nsm_score_threshold
		self.anchor_4dtensors = anchor_4dtensors

		super(PyramidNSMLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		'''
		Arguments
			input_shape: [
				(batch_size, h1, w1, 6k),
				(batch_size, h2, w2, 6k),
				(batch_size, h3, w3, 6k),
				(batch_size, h4, w4, 6k)
			]
		'''

		assert len(input_shape) == 4, 'PyramidNSMLayer must be passed 4 inputs: 4 lavels of clz_tensor & bbe_tensor'

		super(PyramidNSMLayer, self).build(input_shape)

	def compute_output_shape(self, input_shape):
		'''
		Arguments
			input_shape: [
				(batch_size, h1, w1, 6k),
				(batch_size, h2, w2, 6k),
				(batch_size, h3, w3, 6k),
				(batch_size, h4, w4, 6k)
			]

		Return
			None, num_of_rois, 4
		'''

		assert len(input_shape) == 4, 'PyramidNSMLayer must be passed 4 inputs: 4 lavels of clz_tensor & bbe_tensor'
		
		return None, self.num_of_rois, 4
		
	def call(self, x):
		'''
		To compute rois from infered classification branches and location branches
		Arguments:
			x:
		Return
			roi_3dtensor:
		'''

		assert len(x) == 4, 'PyramidNSMLayer must be passed 4 inputs: 4 lavels of clz_tensor & bbe_tensor'

		ishape = self.ishape
		max_num_of_rois = self.num_of_rois
		nsm_iou_threshold = self.nsm_iou_threshold
		anchor_4dtensors = self.anchor_4dtensors
		nsm_score_threshold = self.nsm_score_threshold
		clzbbe_3dtensors = [x[0][0], x[1][0], x[2][0], x[3][0]]

		roi_2dtensor = pnsm(
			anchor_4dtensors=anchor_4dtensors, 
			clzbbe_3dtensors=clzbbe_3dtensors, 
			max_num_of_rois=max_num_of_rois,
			nsm_iou_threshold=nsm_iou_threshold,
			nsm_score_threshold=nsm_score_threshold,
			ishape=ishape) # (num_of_rois, 4)
		roi_3dtensor = tf.expand_dims(input=roi_2dtensor, axis=0) # (batch_size, num_of_rois, 4), batch_size = 1

		return roi_3dtensor








