import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from models import build_test_model
from datagen import genanchors, load_dataset, genbbox, genbbox_com
from datetime import datetime


net_name = "XNet"
output_path = 'output'
test_anno_file_path = '../datasets/widerface/test.txt'
test_image_dir = '../datasets/widerface/test'
ishape = [512, 512, 3] # [64, 64, 3], [128, 128, 3], [256, 256, 3], [512, 512, 3]
combine = True if ishape[0] is 512 else False
ssize = [ishape[0]/4, ishape[1]/4]
asizes = [[32, 32]]
total_classes = 1
resnet_settings = [[8, 8, 32], [24, [2, 2]]]
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 100
total_test_examples = 100

abox_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes), dtype='float32') # (h*w*k, 4)

model = build_test_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	abox_2dtensor=abox_2dtensor,
	nsm_iou_threshold=nsm_iou_threshold,
	nsm_score_threshold=nsm_score_threshold,
	nsm_max_output_size=nsm_max_output_size)
# model.summary()
model.load_weights('{}/weights_best_precision.h5'.format(output_path), by_name=True)
gendata = genbbox_com if combine is True else genbbox
test_dataset = load_dataset(anno_file_path=test_anno_file_path)
gen = gendata(
	dataset=test_dataset, 
	image_dir=test_image_dir, 
	ishape=ishape,
	total_examples=total_test_examples)

for batch in range(total_test_examples):
	image, _ = next(gen)
	batchx_4dtensor = tf.constant(value=[image], dtype='float32')
	print('{}: 1'.format(datetime.now().time()), end='\n')
	boxclz_2dtensor, valid_outputs = model.predict_on_batch(batchx_4dtensor) # (h*w*k, total_classes+1+4)
	print('{}: 2'.format(datetime.now().time()), end='\n')
	total_boxes = int(valid_outputs[0].numpy())
	if total_boxes > 0:
		boxclz_2dtensor = boxclz_2dtensor[:total_boxes]
		pix = batchx_4dtensor[0]

		_, ax = plt.subplots(figsize=(15, 7.35))
		ax.imshow(np.array(pix, dtype='uint8'))

		for i in range(boxclz_2dtensor.shape[0]):
			box = boxclz_2dtensor[i, :4]
			frame = [box[1], box[0], box[3]-box[1], box[2]-box[0]]
			ax.add_patch(Rectangle(
				(frame[0], frame[1]), frame[2], frame[3],
				linewidth=0.5, 
				edgecolor='cyan',
				facecolor='none', 
				linestyle='-'))

		plt.show()
