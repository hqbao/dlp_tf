import tensorflow as tf
from tensorflow.keras.layers import Layer
from utils import nsm

class NSMLayer(Layer):
	'''
	'''
	
	def __init__(self, ishape, num_of_rois, nsm_iou_threshold, nsm_score_threshold, anchor_4dtensor, **kwargs):
		'''
		'''

		self.ishape = ishape
		self.num_of_rois = num_of_rois
		self.nsm_iou_threshold = nsm_iou_threshold
		self.nsm_score_threshold = nsm_score_threshold
		self.anchor_4dtensor = anchor_4dtensor

		super(NSMLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		'''
		Arguments
			input_shape: (batch_size, h, w, 6k)
		'''

		super(NSMLayer, self).build(input_shape)

	def compute_output_shape(self, input_shape):
		'''
		Arguments
			input_shape: (batch_size, h, w, 6k)
		Return
			None, num_of_rois, 4
		'''
		
		return None, self.num_of_rois, 4
		
	def call(self, x):
		'''
		To compute rois from infered classification branch and location branch
		Arguments:
			x:
		Return
			roi_3dtensor:
		'''

		ishape = self.ishape
		anchor_4dtensor = self.anchor_4dtensor
		max_num_of_rois = self.num_of_rois
		nsm_iou_threshold = self.nsm_iou_threshold
		nsm_score_threshold = self.nsm_score_threshold

		clzbbe_4dtensor = x # (batch_size, h, w, 6k)
		clzbbe_3dtensor = clzbbe_4dtensor[0] # (h, w, 6k)

		roi_2dtensor = nsm(
			anchor_4dtensor=anchor_4dtensor,
			clzbbe_3dtensor=clzbbe_3dtensor,
			max_num_of_rois=max_num_of_rois,
			nsm_iou_threshold=nsm_iou_threshold,
			nsm_score_threshold=nsm_score_threshold,
			ishape=ishape)
		roi_3dtensor = tf.expand_dims(input=roi_2dtensor, axis=0) # (batch_size, num_of_rois, 4)

		return roi_3dtensor








