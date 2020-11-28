import cv2
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from skimage import transform


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def detect(interpreter, input_details, output_details, pix):
	'''
	'''
	
	batchx_4dtensor = tf.expand_dims(input=pix, axis=0)
	interpreter.set_tensor(tensor_index=input_details[0]['index'], value=batchx_4dtensor)
	interpreter.invoke()

	valid_outputs = interpreter.get_tensor(output_details[0]['index'])
	detection = interpreter.get_tensor(output_details[1]['index'])

	bbox2d = detection[:int(valid_outputs[0]), :]
	return bbox2d

def detect_landmarks(interpreter, input_details, output_details, x, ishape):
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

	return ya, xa, yb, xb, yc, xc, yd, xd, ye, xe


test_file = 'believer.mp4'
result_file = 'believer_detected.mp4'
output_path = 'output'
ishape = [512, 512, 3]

interpreter = tf.lite.Interpreter(model_path='{}/det_xsml_model.tflite'.format(output_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# pprint(input_details)
# pprint(output_details)

interpreter_ = tf.lite.Interpreter(model_path='{}/ali_model08.tflite'.format(output_path))
interpreter_.allocate_tensors()

input_details_ = interpreter_.get_input_details()
output_details_ = interpreter_.get_output_details()

# pprint(input_details_)
# pprint(output_details_)

cap = cv2.VideoCapture(test_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

processed_images = []

while True:
	# Quit with 'q' press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success, pix = cap.read()
	if success is not True:
		break

	pix = pix[128:-128, :, :]
	h, w, _ = pix.shape
	size = min(h, w)
	origin_y = 0
	origin_x = (w - h)//2
	if h > w:
		origin_y = (h - w)//2
		origin_x = 0
	
	pix = pix[origin_y:origin_y+size, origin_x:origin_x+size, :]
	pix = cv2.resize(pix, (ishape[1], ishape[0]))
	pix = pix/np.max(pix)
	pix = pix*255

	# print('Start: {}'.format(datetime.now().time()))
	imgpix = tf.constant(value=pix, dtype='float32')
	bbox2d = detect(
		interpreter=interpreter, 
		input_details=input_details, 
		output_details=output_details, 
		pix=imgpix)
	# print('End:   {}'.format(datetime.now().time()))

	bboxes = []
	for y1, x1, y2, x2, _ in bbox2d:
		h, w = y2 - y1, x2 - x1
		y, x = int(y1 + 0.5*h), int(x1 + 0.5*w)
		half_edge = int(1.0*max(h, w)/2)
		edge = 2*half_edge
		y1, x1, y2, x2 = int(y-half_edge), int(x-half_edge), int(y+half_edge), int(x+half_edge)

		if y1+half_edge < 0 or x1+half_edge < 0 or y2-half_edge > ishape[0] or x2-half_edge > ishape[1]:
			continue

		iobject = np.zeros((edge, edge, 3))
		_y1, _x1, _y2, _x2 = 0, 0, edge, edge
		__y1, __x1, __y2, __x2 = y1, x1, y2, x2

		if y1 < 0 and y1+half_edge > 0:
			_y1 = -y1
			__y1 = 0

		if x1 < 0 and x1+half_edge > 0:
			_x1 = -x1
			__x1 = 0

		if y2 > ishape[0] and y2-half_edge < ishape[0]:
			_y2 = edge - (y2-ishape[0])
			__y2 = ishape[0]

		if x2 > ishape[1] and x2-half_edge < ishape[1]:
			_x2 = edge - (x2-ishape[1])
			__x2 = ishape[1]

		iobject[_y1:_y2, _x1:_x2, :] = pix[__y1:__y2, __x1:__x2, :]
		iobject = transform.resize(image=iobject, output_shape=[112, 112])
		iobject = np.mean(iobject, axis=-1, keepdims=True)
		iobject = iobject/np.max(iobject)
		iobject = iobject*255
		iobject = np.array(iobject, dtype='int32')

		# ya, xa, yb, xb, yc, xc, yd, xd, ye, xe = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		ya, xa, yb, xb, yc, xc, yd, xd, ye, xe = detect_landmarks(
			interpreter=interpreter_, 
			input_details=input_details_, 
			output_details=output_details_, 
			x=iobject, 
			ishape=[112, 112, 1]) # (5, h, w)

		eye = ya != 0 or yb != 0
		mouth = yc != 0 or yd != 0
		left = ya != 0 or yc != 0
		right = yb != 0 or yd != 0
		left_eye_and_nose = ya != 0 or ye != 0
		right_eye_and_nose = yb != 0 or ye != 0
		left_mouth_and_nose = yc != 0 or ye != 0
		right_mouth_and_nose = yd != 0 or ye != 0
		if eye is not True and mouth is not True and left is not True and right is not True and left_eye_and_nose is not True and right_eye_and_nose is not True and left_mouth_and_nose is not True and right_mouth_and_nose is not True:
			continue

		scale = edge/112
		ya, xa = y1 + int(scale*ya), x1 + int(scale*xa)
		yb, xb = y1 + int(scale*yb), x1 + int(scale*xb)
		yc, xc = y1 + int(scale*yc), x1 + int(scale*xc)
		yd, xd = y1 + int(scale*yd), x1 + int(scale*xd)
		ye, xe = y1 + int(scale*ye), x1 + int(scale*xe)

		bboxes.append([y1, x1, y2, x2, ya, xa, yb, xb, yc, xc, yd, xd, ye, xe])

	for y1, x1, y2, x2, ya, xa, yb, xb, yc, xc, yd, xd, ye, xe in bboxes:
		cv2.circle(pix, (int(x1 + 0.5*(x2 - x1)), int(y1 + 0.5*(y2 - y1))), int(0.5*(y2 - y1)), [255, 255, 255], 1)
		if ya != y1 and yb != y1 and yc != y1 and yd != y1 and ye != y1:
			cv2.circle(pix, (xa, ya), 4, [255, 255, 0], -1)
			cv2.circle(pix, (xb, yb), 4, [255, 255, 0], -1)
			cv2.circle(pix, (xc, yc), 4, [0, 255, 255], -1)
			cv2.circle(pix, (xd, yd), 4, [0, 255, 255], -1)
			cv2.circle(pix, (xe, ye), 4, [255, 128, 255], -1)
	
	# Display frame
	pix = np.array(pix, dtype='uint8')
	cv2.imshow('frame', pix)

	processed_images.append(pix)

cap.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter(result_file, cv2.VideoWriter_fourcc(*'MP4V'), 24, (ishape[1], ishape[0]))
for i in range(len(processed_images)):
    out.write(processed_images[i])
out.release()










