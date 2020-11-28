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
	
	batchx_4dtensor = tf.expand_dims(input=x, axis=0)
	interpreter.set_tensor(tensor_index=input_details[0]['index'], value=batchx_4dtensor)
	interpreter.invoke()

	valid_outputs = interpreter.get_tensor(output_details[0]['index'])
	detection = interpreter.get_tensor(output_details[1]['index'])

	bbox2d = detection[:valid_outputs[0], :]
	return bbox2d


test_file = 'believer.mp4'
result_file = 'believer_detected.mp4'
output_path = 'output'
ishape = [1024, 1024, 3]

interpreter = tf.lite.Interpreter(model_path='{}/det_model.tflite'.format(output_path))
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
	x = tf.constant(value=pix, dtype='float32') # (h, w, 3)
	bbox2d = detect(interpreter=interpreter, x=x, ishape=ishape)
	# print('End: {}'.format(datetime.now().time()))

	for y1, x1, y2, x2, _ in bbox2d:
		cv2.rectangle(pix, (x1, y1), (x2, y2), [255, 255, 0], 1)
	
	# Display frame
	cv2.imshow('frame', np.array(pix, dtype='uint8'))

	processed_images.append(pix)

cap.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter(result_file, cv2.VideoWriter_fourcc(*'MP4V'), 24, (ishape[1], ishape[0]))
for i in range(len(processed_images)):
    out.write(processed_images[i])
out.release()


