import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_model
from datagen import load_dataset, genbbox, genbbox_com
from datetime import datetime


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mode = 'test'
anno_file_path = '../datasets/widerface/'+mode+'.txt'
image_dir = '../datasets/widerface/'+mode
ishape = [512, 512, 3] # [64, 64, 3], [128, 128, 3], [256, 256, 3], [512, 512, 3]
combine = True if ishape[0] is 512 else False
total_examples = 100

gendata = genbbox_com if combine is True else genbbox
dataset = load_dataset(anno_file_path=anno_file_path)
gen = gendata(
	dataset=dataset, 
	image_dir=image_dir, 
	ishape=ishape,
	total_examples=total_examples)

for _ in range(total_examples):
	# print('{}: 1'.format(datetime.now().time()), end='\n')
	pix, bboxes = next(gen)
	# print('{}: 2'.format(datetime.now().time()), end='\n')

	_, ax = plt.subplots(figsize=(15, 7.35))
	ax.imshow(np.array(pix, dtype='uint8'))

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

