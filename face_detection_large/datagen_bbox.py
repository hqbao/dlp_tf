import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_model
from datagen import load_dataset, genbbox
from datetime import datetime


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mode = 'train'
anno_file_path = '../datasets/widerface/'+mode+'.txt'
image_dir = '../datasets/widerface/'+mode
ishape = [64, 64, 3]
total_examples = 100

dataset = load_dataset(anno_file_path=anno_file_path)
gen = genbbox(
	dataset=dataset, 
	image_dir=image_dir, 
	ishape=ishape,
	total_examples=total_examples)

for _ in range(total_examples):
	# print('{}: 1'.format(datetime.now().time()), end='\n')
	pix, bboxes = next(gen)
	# print('{}: 2'.format(datetime.now().time()), end='\n')

	_, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(np.array(pix, dtype='int32'))

	for i in range(len(bboxes)):
		box = bboxes[i][:4]
		frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
		ax.add_patch(Rectangle(
			(frame[0], frame[1]), frame[2], frame[3],
			linewidth=0.8, 
			edgecolor='cyan',
			facecolor='none', 
			linestyle='-'))

	plt.show()














