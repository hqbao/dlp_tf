import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from datagen import genx, gety, gendy
from utils import pad_roi2d, bbe2box2d, box2frame


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

start_example_index = 0
num_of_examples = 100

asizes = [	
	[[32, 32]],
	[[64, 64]],
	[[128, 128]],
	[[256, 256]],
]
ishape = [1024, 1024, 3]
feature_map_sizes = [[128, 128], [64, 64], [32, 32], [16, 16]]
frame_mode = True
classes = ['face', 'non-object']
mapping = {0: 0}
iou_thresholds = [0.3, 0.5]
max_num_of_rois = 20
unified_roi_size = [7, 7]
ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

gen = genx(
	coco=coco, 
	img_dir=img_dir, 
	classes=classes, 
	limit=[start_example_index, start_example_index+num_of_examples],
	ishape=ishape)

for batch_idx in range(num_of_examples):

	# generate x
	x, img_id = next(gen)

	# get labels
	bbox2d, tmasks = gety(coco=coco, img_id=img_id, classes=classes, frame_mode=frame_mode, mapping=mapping)

	# predict rois
	roi2d = pad_roi2d(roi2d=bbox2d[:, :4], max_num_of_rois=max_num_of_rois, pad_roi=[0, 0, ishape[0], ishape[1]])
	roi_2dtensor = tf.constant(value=roi2d, dtype='float32')

	clzbbe_2dtensor = gendy(
		num_of_classes=len(classes), 
		roi_2dtensor=roi_2dtensor, 
		bbox2d=bbox2d, 
		unified_roi_size=unified_roi_size,
		isize=ishape[:2], 
		iou_thresholds=iou_thresholds)

	clz_2dtensor = clzbbe_2dtensor[:, :len(classes)]
	bbe_2dtensor = clzbbe_2dtensor[:, len(classes):]

	pred_bbox2d = bbe2box2d(box_2dtensor=roi_2dtensor, bbe_2dtensor=bbe_2dtensor)

	fig, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(x/255)

	for k in range(roi_2dtensor.shape[0]):
		clz1d = np.array(clz_2dtensor[k]) # onehot
		if np.sum(clz1d) == 0:
			continue

		clz = np.argmax(clz1d)
		pred_bbox = pred_bbox2d[k]
		tframe = box2frame(box=pred_bbox, apoint=[0.5, 0.5])

		ax.annotate(classes[clz], (tframe[0], tframe[1]), color='green', weight='bold', fontsize=8, ha='center', va='center')

		tframe = box2frame(box=pred_bbox, apoint=[0.0, 0.0])

		ax.add_patch(Rectangle(
			(tframe[0], tframe[1]), tframe[2], tframe[3],
			linewidth=1, 
			edgecolor='g',
			facecolor='none', 
			linestyle='-'))

	plt.show()








