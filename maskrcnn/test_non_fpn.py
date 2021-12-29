import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import random
from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from utils import genanchors, box2frame
from models import build_inference_maskrcnn_non_fpn


def resize(pix, ishape):
	'''
	'''

	scale = pix.shape[1]/ishape[1]
	cropped_y = int(pix.shape[1]*(ishape[0]/ishape[1]))
	cropped_x = pix.shape[1]
	pix = pix[:cropped_y, :cropped_x, :]
	pix = cv2.resize(pix, (ishape[1], ishape[0]))
	return pix


def detect(pix, detection_model):
	'''
	'''
	
	x = pix
	x = tf.constant(value=x, dtype='float32')
	batch_x = tf.expand_dims(input=x, axis=0)
	box_2dtensor = detection_model.predict_on_batch(batch_x)
	box2d = np.array(box_2dtensor)
	return box2d

def cacl_rgb(num):
	'''
	'''

	quo = num
	r = quo%256
	quo = quo//256
	g = quo%256
	b = quo//256
	return r, g, b

def is_static_object(route, min_area_of_static_object):
	'''
	'''

	min_y = 32768 # 2^15
	min_x = 32768
	max_y = 0
	max_x = 0
	for x, y, _, _ in route:
		if min_y > y:
			min_y = y
		if min_x > x:
			min_x = x
		if max_y < y:
			max_y = y
		if max_x < x:
			max_x = x

	return max_y - min_y < min_area_of_static_object and max_x - min_x < min_area_of_static_object

def in_box(point, box):
	'''
	'''

	x, y = point
	y1, x1, y2, x2 = box
	return y >= y1 and y <= y2 and x >= x1 and x <= x2

def update(routes, boxes, clean_after, min_frame_of_pos_route):
	'''
	To do:
		- Add boxes to the suitable routes
		- Remmove invalid routes
	Arguments
		routes: list of integers and tuples, tuples are points
		boxes:
		clean_after:
		min_frame_of_pos_route:
	Note
		16777216 = 256*256*256 which is r*g*b
	'''

	if len(boxes) == 0:
		return

	if len(routes) == 0:
		for box in boxes:
			routes.append([random.randint(0, 16777216), box2frame(box)])
		return

	added_route_indices = []

	for box in boxes:
		frame = box2frame(box)
		added = False
		removed = []

		for i in range(len(routes)):

			if i in added_route_indices:
				continue

			route = routes[i]
			last_item = route[-1]
			if type(last_item) is int:
				if last_item == -1:
					continue

				if last_item > clean_after:
					removed.append(i)
					continue

				routes[i][-1] += 1
				last_item = route[-2]

			if in_box(point=last_item[:2], box=box):
				routes[i].append(frame)
				added_route_indices.append(i)
				added = True
			else:
				if type(route[-1]) is int:
					routes[i][-1] += 1
				else:
					routes[i].append(1)

		# Remove invalid route after some frames
		for idx in sorted(removed, reverse=True):
			if len(routes[idx]) < min_frame_of_pos_route:
				del routes[idx]
			else:
				routes[idx].append(-1) #no more adding for this route

		if added is not True:
			routes.append([random.randint(0, 16777216), frame])


ishape 						= [1024, 1024, 3]
ssize 						= [32, 32]
asizes						= [[91, 181], [128, 128], [181, 91]]
output_path					= 'output'
test_file					= '../test.mp4'
result_file					= '../result.mp4'
classes 					= ['face', 'none']
block_settings				= [[4, 4, 16], [2, [2, 2]], [3, [1, 1]], [4, [1, 1]], [5, [1, 1]]]
max_num_of_rois 			= 7
nsm_iou_threshold 			= 0.2
nsm_score_threshold			= 0.1
unified_roi_size			= [3, 3]
rpn_head_dim				= 64
fc_denses					= [8]
colors						= [[255, 255, 0]]
min_frame_of_pos_route		= 0
min_area_of_static_object	= 0
clean_after					= 128 # frames
box_pad						= 0 # pixels
entrance					= [200, 200, 270, 400]


abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
anchor_4dtensor = tf.constant(value=abox4d, dtype='float32')

rpn_model, detection_model = build_inference_maskrcnn_non_fpn(
	ishape=ishape, 
	anchor_4dtensor=anchor_4dtensor, 
	classes=classes, 
	max_num_of_rois=max_num_of_rois, 
	nsm_iou_threshold=nsm_iou_threshold, 
	nsm_score_threshold=nsm_score_threshold, 
	unified_roi_size=unified_roi_size,
	rpn_head_dim=rpn_head_dim,
	fc_denses=fc_denses,
	block_settings=block_settings,
	base_block_trainable=False)

rpn_model.load_weights('{}/rpn_weights.h5'.format(output_path), by_name=True)
detection_model.load_weights('{}/detection_weights.h5'.format(output_path), by_name=True)

cap = cv2.VideoCapture(test_file)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

processed_images = []
routes = []

while True:
	# Quit with 'q' press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success, pix = cap.read()

	if success is not True:
		break

	# Resize origin frame to model input
	pix = resize(pix=pix, ishape=ishape)

	# Execute detection
	box2d = detect(pix=pix, detection_model=detection_model)
	boxes = list(box2d)
	boxes = [box[4:8] for box in boxes if (box[0] != 0 or box[1] != 0 or box[2] != ishape[0] or box[3] != ishape[1]) and box[8] != len(classes)-1]

	if len(boxes) == 0:
		boxes = [[0, 0, 0, 0]]

	# Display entrance box
	cv2.rectangle(pix, (entrance[1], entrance[0]), (entrance[3], entrance[2]), [255, 0, 0], 1)

	# Display objects detected
	for box in boxes:
		cv2.rectangle(pix, (box[1], box[0]), (box[3], box[2]), [255, 255, 255], 1)

	# Make boxes smaller
	box2d = np.array(boxes, dtype='float32')
	box2d[:, 0:2] += box_pad
	box2d[:, 2:4] -= box_pad
	boxes = list(box2d)

	# Put new boxes into the route list and clean invalid routes
	update(
		routes=routes, 
		boxes=boxes, 
		clean_after=clean_after, 
		min_frame_of_pos_route=min_frame_of_pos_route)

	up = 0
	down = 0

	for route in routes:
		r, g, b = cacl_rgb(route[0])
		route = [p for p in route[1:] if type(p) is list]
		route_len = len(route)

		# Terminals
		start_point = np.array(route[0][:2], dtype='int')
		end_point = np.array(route[-1][:2], dtype='int')
		end_frame = np.array(route[-1], dtype='int')

		# Check if not static detection
		if is_static_object(route=route, min_area_of_static_object=min_area_of_static_object):
			continue

		# Check if a person moving
		if route_len < min_frame_of_pos_route:
			continue

		# # Display object box
		# box = [
		# 	int(end_frame[1]-0.5*(end_frame[3])),
		# 	int(end_frame[0]-0.5*(end_frame[2])),
		# 	int(end_frame[1]+0.5*(end_frame[3])),
		# 	int(end_frame[0]+0.5*(end_frame[2]))
		# ]
		# cv2.rectangle(pix, (box[1], box[0]), (box[3], box[2]), [255, 255, 255], 1)

		IN, OUT = False, False

		# A person is IN when route starts inside entrance and ends outside entrance
		if in_box(start_point, entrance) and in_box(end_point, entrance) is not True:
			IN = True

		# A person is OUT when route starts output entrance and ends inside entrance
		if in_box(start_point, entrance) is not True and in_box(end_point, entrance):
			OUT = True

		if (IN and OUT) or (not IN and not OUT):
			continue

		up += int(IN)
		down += int(OUT)

		# Display route
		cv2.circle(pix, (start_point[0], start_point[1]), 6, [r, g, b], -1) # start point
		prev_point = start_point
		for item in route[1:-2]:
			item = np.array(item, dtype='int')
			cv2.line(pix, (prev_point[0], prev_point[1]), (item[0], item[1]), [r, g, b], 1)
			prev_point = item
		cv2.circle(pix, (end_point[0], end_point[1]), 4, [r, g, b], 2) # end point

	# Display in count
	cv2.putText(pix, 'UP: {}'.format(up), (8, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)
	cv2.putText(pix, 'DOWN: {}'.format(down), (8, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)
		
	# Display frame
	cv2.imshow('frame', pix)

	processed_images.append(pix)

cap.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter(result_file, cv2.VideoWriter_fourcc(*'MP4V'), 30, (ishape[1], ishape[0]))
for i in range(len(processed_images)):
    out.write(processed_images[i])
out.release()





