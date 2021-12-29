import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import cv2

from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from utils import genanchors, box2frame


start_example_index = 0
num_of_examples = 100

asizes = [[91, 181], [128, 128], [181, 91]]
ishape = [1024, 1024, 3]
feature_map_size = [32, 32]

all_anchor_points = False
anchor_points = [
	[3, 5],
	[21, 5],
	[26, 56],
	[3, 27],
	[30, 17],
	[16, 32]
]

classes = ['face', 'none']
ann_file = '../datasets/coco/annotations/instances_face.json'
img_dir = '../datasets/coco/images/face'
coco = COCO(ann_file)

cat_ids = coco.getCatIds(catNms=classes)
img_ids = coco.getImgIds(catIds=cat_ids)
imgs = coco.loadImgs(img_ids)

for img in imgs[start_example_index:start_example_index+num_of_examples]:

	# image data (h, w, channels)
	pix = io.imread('{}/{}'.format(img_dir, img['file_name']))

	# padding input img 
	x = np.zeros(ishape, dtype='int32')
	if len(pix.shape) == 2:
		x[:pix.shape[0], :pix.shape[1], 0] = pix
		x[:pix.shape[0], :pix.shape[1], 1] = pix
		x[:pix.shape[0], :pix.shape[1], 2] = pix
	else:
		x[:pix.shape[0], :pix.shape[1], :] = pix

	# generate anchor boxes
	aboxes = genanchors(
		isize=ishape[:2],
		ssize=feature_map_size,
		asizes=asizes)

	fig, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(x/255)
	
	for i in range(aboxes.shape[0]):
		for j in range(aboxes.shape[1]):
			for k in range(aboxes.shape[2]):
				if all_anchor_points:

					abox = aboxes[i, j, k]
					frame = box2frame(box=abox, apoint=[0, 0])

					ax.add_patch(Rectangle(
						(frame[0], frame[1]), frame[2], frame[3],
						linewidth=1, 
						edgecolor='r',
						facecolor='none', 
						linestyle='-'))
				else:
					for point in anchor_points:
						if i == point[0] and j == point[1]:

							abox = aboxes[i, j, k]
							frame = box2frame(box=abox, apoint=[0, 0])

							ax.add_patch(Rectangle(
								(frame[0], frame[1]), frame[2], frame[3],
								linewidth=1, 
								edgecolor='r',
								facecolor='none', 
								linestyle='-'))

	plt.show()





