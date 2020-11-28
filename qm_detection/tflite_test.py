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

	bbox2d = detection[:int(valid_outputs[0]), :]
	return bbox2d

def calc_score(pred_bbox2d, true_answer):
	'''
	'''

	box2d = pred_bbox2d
	if box2d.shape[0] != 250:
		return None, None, None

	boxes = list(box2d)
	boxes.sort(key=lambda box: box[0])
	box2d = np.array(boxes)

	up_box2d = box2d[:130]
	for i in range(10):
		boxes = list(up_box2d[i*13:i*13+13])
		boxes.sort(key=lambda box: box[1])
		up_box2d[i*13:i*13+13, :] = np.array(boxes)

	up_box2d = up_box2d.reshape((10, 13, -1))
	up_box2d = up_box2d[:, :, 4]

	examinee_code_matrix = up_box2d[:, :6]
	answer_code_matrix = up_box2d[:, 6:9]
	answer1_matrix = up_box2d[:, 9:]

	down_box2d = box2d[130:]
	for i in range(10):
		boxes = list(down_box2d[i*12:i*12+12])
		boxes.sort(key=lambda box: box[1])
		down_box2d[i*12:i*12+12, :] = np.array(boxes)

	down_box2d = down_box2d.reshape((10, 12, -1))
	down_box2d = down_box2d[:, :, 4]
	
	answer2_matrix = down_box2d[:, :4]
	answer3_matrix = down_box2d[:, 4:8]
	answer4_matrix = down_box2d[:, 8:]

	examinee_code = np.argmin(examinee_code_matrix, axis=0)
	# print("examinee_code: ", end='')
	# print(examinee_code)

	answer_code = np.argmin(answer_code_matrix, axis=0)
	# print("answer_code: ", end='')
	# print(answer_code)

	answer = np.concatenate([answer1_matrix, answer2_matrix, answer3_matrix, answer4_matrix], axis=0)
	# print("answer: ", end='')
	# print(answer)

	answer = np.argmin(answer, axis=1)
	score = np.where(answer==true_answer, 1, 0)
	score = np.sum(score)

	return examinee_code, answer_code, score


test_file = 'qm_input.mp4'
result_file = 'qm_output.mp4'
output_path = 'output'
ishape = [240, 200, 3]
true_answer = np.ones(40)

interpreter = tf.lite.Interpreter(model_path='{}/model.tflite'.format(output_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# pprint(input_details)
# pprint(output_details)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

processed_images = []
scale = ishape[0]/ishape[1]
str1 = ''
str2 = ''
str3 = ''

while True:
	# Quit with 'q' press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success, pix = cap.read()
	if success is not True:
		break

	h, w, _ = pix.shape
	y = h//2
	x = w//2
	size = min(h, w)

	origin_y = y - size//2
	origin_x = x - int(size/scale)//2
	
	pix = pix[origin_y:origin_y+size, origin_x:origin_x+int(size/scale), :]
	pix = cv2.resize(pix, (ishape[1], ishape[0]))

	# Bright up
	pix = np.clip(pix, 16, 255-16)
	pix = pix/np.max(pix)
	pix = pix*255

	# print('Start: {}'.format(datetime.now().time()))
	x = tf.constant(value=pix, dtype='float32') # (h, w, 3)
	bbox2d = detect(interpreter=interpreter, x=x, ishape=ishape)
	# print('End: {}'.format(datetime.now().time()))

	# Scoring
	examinee_code, answer_code, score = calc_score(pred_bbox2d=bbox2d, true_answer=true_answer)
	if score is not None:
		print(examinee_code, answer_code, score)
		str1 = ''.join(list(map(str, examinee_code)))
		str2 = ''.join(list(map(str, answer_code)))
		str3 = str(score)

	cv2.putText(pix, 'EXAMINEE CODE: {}'.format(str1), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 255, 0], 1)
	cv2.putText(pix, 'ANSWER CODE: {}'.format(str2), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 255, 0], 1)
	cv2.putText(pix, 'SCORE: {}'.format(str3), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 255, 0], 1)

	for y1, x1, y2, x2, cat in bbox2d:
		color = [255, 255, 0] if cat == 0 else [255, 255, 255]
		cv2.rectangle(pix, (x1, y1), (x2, y2), color, 1)
	
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


