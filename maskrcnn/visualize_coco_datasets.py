import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import tensorflow as tf

from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
from utils import box2frame
from datagen import genx, gety

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

ishape = [1024, 1024, 3]
frame_mode = True
mapping = {0: 0}

classes = ['face', 'none']ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

# cat_ids = coco.getCatIds(catNms=classes)
gen = genx(coco=coco, img_dir=img_dir, classes=classes, limit=[0, 100], ishape=ishape)

for i in range(100):
	# generate x
	x, img_id = next(gen)

	bbox2d, _ = gety(coco, img_id, classes, frame_mode=frame_mode, mapping=mapping)

	fig, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(x/255)

	for bbox in bbox2d:
		frame = box2frame(box=bbox, apoint=[0, 0])
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3], 
			linewidth=1, 
			edgecolor='yellow',
			facecolor='none', 
			linestyle='-'))

	# ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=0)
	# anns = coco.loadAnns(ids=ann_ids)
	# coco.showAnns(anns)

	plt.show()




