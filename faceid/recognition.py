import cv2
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from skimage import transform
from utils import detect_faces, detect_landmarks, generate_embedding, recognize


output_path = 'outputs'
test_file = 'videos/ongiocaudayroi1.mp4'
result_file = output_path+'/ongiocaudayroi1_detected.mp4'
tflite_model_path = 'tflite_models'
ishape = [512, 512, 3]
ID_map = [
'shark_do_thi_kim_lien', 'phung_dang_khoa', 'tu_long', 'manh_quynh', 'huong_giang', 'luxstay_van_dung', 'mi_du', 'truong_giang', 'map_vai_phu', 'vo_hoang_yen', 'hoang_thuy_linh', 'dieu_nhi', 'erik', 'le_duong_bao_lam', 'xuan_bac', 'lan_ngoc', 'tran_thanh', 'shark_nguyen_ngoc_thuy', 'duc_phuc', 'hari_won', 'tien_luat', 'tuan_tran', 'shark_pham_thanh_hung', 'miu_le', 'chi_tai', 'shark_nguyen_manh_dung', 'viet_huong', 'le_giang', 'le_tin', 'hong_kim_hanh', 'hoai_linh', 'vi_da', 'shark_nguyen_thanh_viet', 'linh_ngoc_dam']

interpreter1 = tf.lite.Interpreter(model_path=tflite_model_path+'/det_xsml_model.tflite')
interpreter1.allocate_tensors()

input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()
# pprint(input_details1)
# pprint(output_details1)

interpreter2 = tf.lite.Interpreter(model_path=tflite_model_path+'/ali_model08.tflite')
interpreter2.allocate_tensors()

input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

# pprint(input_details2)
# pprint(output_details2)

interpreter3 = tf.lite.Interpreter(model_path=tflite_model_path+'/rec_model.tflite')
interpreter3.allocate_tensors()

input_details3 = interpreter3.get_input_details()
output_details3 = interpreter3.get_output_details()

# pprint(input_details2)
# pprint(output_details2)

cap = cv2.VideoCapture(test_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, 10000)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

processed_images = []
embedding2d = np.load(output_path+'/embedding2d.npy')

while True:
	# Quit with 'q' press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success, pix = cap.read()
	if success is not True:
		break

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
		interpreter=interpreter1, 
		input_details=input_details1, 
		output_details=output_details1, 
		pix=imgpix)
	# print('End:   {}'.format(datetime.now().time()))

	bboxes = []
	for y1, x1, y2, x2, _ in bbox2d:
		h, w = y2 - y1, x2 - x1
		y, x = int(y1 + 0.5*h), int(x1 + 0.5*w)
		half_edge = int(1.1*max(h, w)/2)
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
		iobject = transform.resize(image=iobject, output_shape=[134, 134])
		_iobject = np.mean(iobject, axis=-1, keepdims=True)
		_iobject = _iobject/np.max(_iobject)
		_iobject = _iobject*255
		_iobject = np.array(_iobject, dtype='int32')
		_iobject = _iobject[11:-11, 11:-11, :]

		ya, xa, yb, xb, yc, xc, yd, xd, ye, xe = detect_landmarks(
			interpreter=interpreter2, 
			input_details=input_details2, 
			output_details=output_details2, 
			pix=_iobject, 
			ishape=[112, 112, 1]) # (5, h, w)

		likely_face = False
		eye = ya != 0 or yb != 0
		mouth = yc != 0 or yd != 0
		left = ya != 0 or yc != 0
		right = yb != 0 or yd != 0
		left_eye_and_nose = ya != 0 or ye != 0
		right_eye_and_nose = yb != 0 or ye != 0
		left_mouth_and_nose = yc != 0 or ye != 0
		right_mouth_and_nose = yd != 0 or ye != 0
		if eye is True or mouth is True or left is True or right is True or left_eye_and_nose is True or right_eye_and_nose is True or left_mouth_and_nose is True or right_mouth_and_nose is True:
			likely_face = True

		name = ''
		iobject = transform.resize(image=iobject, output_shape=[112, 112])
		iobject = np.mean(iobject, axis=-1, keepdims=True)
		iobject = np.concatenate([iobject, iobject, iobject], axis=-1)
		iobject = np.array(iobject, dtype='int32')
		embedding1d = generate_embedding(
			interpreter=interpreter3, 
			input_details=input_details3, 
			output_details=output_details3, 
			pix=iobject)
		oid = recognize(embedding2d=embedding2d, embedding1d=embedding1d)
		name = '' if oid == -1 else ID_map[oid]

		if likely_face is not True and name == '':
			continue

		scale = edge/112
		ya, xa = y1 + int(scale*ya), x1 + int(scale*xa)
		yb, xb = y1 + int(scale*yb), x1 + int(scale*xb)
		yc, xc = y1 + int(scale*yc), x1 + int(scale*xc)
		yd, xd = y1 + int(scale*yd), x1 + int(scale*xd)
		ye, xe = y1 + int(scale*ye), x1 + int(scale*xe)

		bboxes.append([y1, x1, y2, x2, ya, xa, yb, xb, yc, xc, yd, xd, ye, xe, name])

	for y1, x1, y2, x2, ya, xa, yb, xb, yc, xc, yd, xd, ye, xe, name in bboxes:
		cv2.circle(pix, (int(x1 + 0.5*(x2 - x1)), int(y1 + 0.5*(y2 - y1))), int(0.5*(y2 - y1)), [255, 255, 255], 1)

		if ya != y1 and yb != y1 and yc != y1 and yd != y1 and ye != y1:
			cv2.circle(pix, (xa, ya), 4, [255, 255, 0], -1)
			cv2.circle(pix, (xb, yb), 4, [255, 255, 0], -1)
			cv2.circle(pix, (xc, yc), 4, [0, 255, 255], -1)
			cv2.circle(pix, (xd, yd), 4, [0, 255, 255], -1)
			cv2.circle(pix, (xe, ye), 4, [255, 128, 255], -1)

		cv2.putText(pix, name, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 0], 1)
	
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

