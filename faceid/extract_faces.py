import cv2
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from skimage import transform
from utils import detect_faces, detect_landmarks


test_file = 'videos/sharktank1.mp4'
tflite_model_path = 'tflite_models'
output_path = 'outputs'
ishape = [512, 512, 3]
step_size = 16

interpreter = tf.lite.Interpreter(model_path=tflite_model_path+'/det_xsml_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# pprint(input_details)
# pprint(output_details)

interpreter_ = tf.lite.Interpreter(model_path=tflite_model_path+'/ali_model08.tflite')
interpreter_.allocate_tensors()

input_details_ = interpreter_.get_input_details()
output_details_ = interpreter_.get_output_details()

# pprint(input_details_)
# pprint(output_details_)

cap = cv2.VideoCapture(test_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

counter = 0
face_idx = 0

while True:
	# Quit with 'q' press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success, pix = cap.read()
	if success is not True:
		break

	cap.set(cv2.CAP_PROP_POS_FRAMES, counter*step_size)
	counter += 1

	pix = pix[:, :, :]
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
	bbox2d = detect_faces(
		interpreter=interpreter, 
		input_details=input_details, 
		output_details=output_details, 
		pix=imgpix)
	# print('End:   {}'.format(datetime.now().time()))

	bboxes = []
	for y1, x1, y2, x2, _ in bbox2d:
		h, w = y2 - y1, x2 - x1
		y, x = y1 + 0.5*h, x1 + 0.5*w
		edge = 1.0*max(h, w)
		y1, x1, y2, x2 = int(y - 0.5*edge), int(x - 0.5*edge), int(y + 0.5*edge), int(x + 0.5*edge)

		if y1 < 0 or x1 < 0 or y2 > ishape[0] or x2 > ishape[1]:
			continue

		iobject = pix[y1:y2, x1:x2, :]
		iobject = transform.resize(image=iobject, output_shape=[112, 112])
		igreyobject = np.mean(iobject, axis=-1, keepdims=True)
		igreyobject = igreyobject/np.max(igreyobject)
		igreyobject = igreyobject*255
		igreyobject = np.array(igreyobject, dtype='int32')

		ya, xa, yb, xb, yc, xc, yd, xd, ye, xe = detect_landmarks(
			interpreter=interpreter_, 
			input_details=input_details_, 
			output_details=output_details_, 
			pix=igreyobject, 
			ishape=[112, 112, 1]) # (5, h, w)

		wider_iobject = pix[y1-int(0.2*(y2-y1)):y2+int(0.2*(y2-y1)), x1-int(0.2*(x2-x1)):x2+int(0.2*(x2-x1)), :]
		if ya != 0 and yb != 0 and yc != 0 and yd != 0 and ye != 0 and wider_iobject.shape[0] == wider_iobject.shape[1] and wider_iobject.shape[0] != 0:
			wider_iobject = transform.resize(image=wider_iobject, output_shape=[112, 112])
			cv2.imwrite(output_path+'/objects/'+str(face_idx)+'.jpg', np.array(wider_iobject, dtype='uint8'))
			face_idx += 1

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

cap.release()
cv2.destroyAllWindows()
