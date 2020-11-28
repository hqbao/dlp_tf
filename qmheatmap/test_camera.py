import cv2
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def detect(interpreter, x, ishape):
	'''
	'''
	
	batch_x = tf.expand_dims(input=x, axis=0)
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

	return [ya, xa, yb, xb, yc, xc, yd, xd, ye, xe]


test_file = '/Users/baulhoa/Desktop/IMG_7760.MOV'
result_file = '/Users/baulhoa/Desktop/bao.mp4'
output_path = 'output'
ishape = [112, 112, 1]

interpreter = tf.lite.Interpreter(model_path='{}/ali/ali_model.tflite'.format(output_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# pprint(input_details)
# pprint(output_details)

cap = cv2.VideoCapture(test_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

processed_images = []

while True:
	# Quit with 'q' press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success, pix = cap.read()
	if success is not True:
		break

	pix = pix.transpose([1, 0, 2]) # for MOV

	h = w = pix.shape[1]
	origin_y = (pix.shape[0] - h)//2
	pix = pix[origin_y:origin_y+h, :, :]
	pix = cv2.resize(pix, (ishape[1], ishape[0]))

	print('Start: {}'.format(datetime.now().time()))
	x = tf.constant(value=pix, dtype='int32') # (h, w, 3)
	x = tf.reduce_sum(input_tensor=x, axis=-1, keepdims=True) # (h, w, 1)
	x = x//3  # (h, w, 1)
	[ya, xa, yb, xb, yc, xc, yd, xd, ye, xe] = detect(interpreter=interpreter, x=x, ishape=ishape) # (5, h, w)
	print('End: {}'.format(datetime.now().time()))
	
	pix[ya-1:ya+1, xa-1:xa+1, :] = 255
	pix[yb-1:yb+1, xb-1:xb+1, :] = 255
	pix[yc-1:yc+1, xc-1:xc+1, :] = 255
	pix[yd-1:yd+1, xd-1:xd+1, :] = 255
	pix[ye-1:ye+1, xe-1:xe+1, :] = 255
	
	# Display frame
	cv2.imshow('frame', pix)

	processed_images.append(pix)

cap.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter(result_file, cv2.VideoWriter_fourcc(*'MP4V'), 30, (ishape[1], ishape[0]))
for i in range(len(processed_images)):
    out.write(processed_images[i])
out.release()


