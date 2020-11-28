import tensorflow as tf
import numpy as np


def detect_faces(interpreter, input_details, output_details, pix):
	'''
	'''
	
	batchx_4dtensor = tf.expand_dims(input=pix, axis=0)
	interpreter.set_tensor(tensor_index=input_details[0]['index'], value=batchx_4dtensor)
	interpreter.invoke()

	valid_outputs = interpreter.get_tensor(output_details[0]['index'])
	detection = interpreter.get_tensor(output_details[1]['index'])

	bbox2d = detection[:int(valid_outputs[0]), :]
	return bbox2d

def detect_landmarks(interpreter, input_details, output_details, pix, ishape):
	'''
	'''
	
	batch_x = tf.expand_dims(input=pix, axis=0)
	interpreter.set_tensor(tensor_index=input_details[0]['index'], value=batch_x)
	interpreter.invoke()

	a = interpreter.get_tensor(output_details[0]['index']) # (5, h, w), uint8
	b = interpreter.get_tensor(output_details[1]['index']) # (5, h, w), uint8
	c = interpreter.get_tensor(output_details[2]['index']) # (5, h, w), uint8
	d = interpreter.get_tensor(output_details[3]['index']) # (5, h, w), uint8
	e = interpreter.get_tensor(output_details[4]['index']) # (5, h, w), uint8

	ya = int(a//ishape[0])
	xa = int(a%ishape[0])
	yb = int(b//ishape[0])
	xb = int(b%ishape[0])
	yc = int(c//ishape[0])
	xc = int(c%ishape[0])
	yd = int(d//ishape[0])
	xd = int(d%ishape[0])
	ye = int(e//ishape[0])
	xe = int(e%ishape[0])
	return ya, xa, yb, xb, yc, xc, yd, xd, ye, xe

def generate_embedding(interpreter, input_details, output_details, pix):
	'''
	'''
	
	batchx = tf.expand_dims(input=pix, axis=0)
	interpreter.set_tensor(tensor_index=input_details[0]['index'], value=batchx)
	interpreter.invoke()
	embedding = interpreter.get_tensor(output_details[0]['index'])[0]
	return embedding

def recognize(embedding2d, embedding1d):
	'''
	'''

	distances = np.zeros(embedding2d.shape[0])
	for i in range(embedding2d.shape[0]):
		distances[i] = tf.norm(tensor=embedding2d[i]-embedding1d)

	min_distance = np.min(distances)
	object_idx = -1 if min_distance > 0.5 else np.argmin(distances)

	return object_idx

