import tensorflow as tf
import numpy as np
from models import build_model, nsm
from datagen import genanchors, comiou, load_dataset, genxy, genxy_com


net_name = "XNet"
output_path = 'output/xsmall'
train_anno_file_path = '../datasets/widerface/train.txt'
train_image_dir = '../datasets/widerface/train'
test_anno_file_path = '../datasets/widerface/test.txt'
test_image_dir = '../datasets/widerface/test'
ishape = [512, 512, 3] # [64, 64, 3], [128, 128, 3], [256, 256, 3], [512, 512, 3]
combine = True if ishape[0] == 512 else False
ssize = [ishape[0]/4, ishape[1]/4]
asizes = [[32, 32]]
total_classes = 1
resnet_settings = [[8, 8, 32], [24, [2, 2]]]
iou_thresholds = [0.3, 0.5]
anchor_sampling = 256
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
total_epoches = 1000
nsm_max_output_size = 100
total_train_examples = 4
total_test_examples = 4

abox_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes), dtype='float32') # (h*w*k, 4)

model = build_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	k=len(asizes), 
	total_classes=total_classes,
	net_name=net_name)
# model.summary()
model.load_weights('{}/weights_.h5'.format(output_path), by_name=True)

min_loss = 2**32
max_precision = 0
max_recall = 0
max_precision_recall = 0

gendata = genxy_com if combine is True else genxy
train_dataset = load_dataset(anno_file_path=train_anno_file_path)
test_dataset = load_dataset(anno_file_path=test_anno_file_path)

for epoch in range(total_epoches):
	# tf.keras.backend.set_value(model.optimizer.learning_rate, 0.001)

	gen = gendata(
		dataset=train_dataset, 
		image_dir=train_image_dir, 
		ishape=ishape, 
		abox_2dtensor=abox_2dtensor, 
		iou_thresholds=iou_thresholds, 
		total_examples=total_train_examples,
		total_classes=total_classes, 
		anchor_sampling=anchor_sampling)

	print('\nTrain epoch {}'.format(epoch))
	loss = np.zeros(total_train_examples)

	for batch in range(total_train_examples):
		batchx_4dtensor, batchy_2dtensor, _ = next(gen)
		batch_loss = model.train_on_batch(batchx_4dtensor, batchy_2dtensor)
		loss[batch] = batch_loss

		print('-', end='')
		if batch%100==99:
			print('{:.2f}%'.format((batch+1)*100/total_train_examples), end='\n')

	mean_loss = float(np.mean(loss, axis=-1))
	print('\nLoss: {:.3f}'.format(mean_loss))

	model.save_weights('{}/weights_.h5'.format(output_path))

	print('\nValidate')

	gen = gendata(
		dataset=test_dataset, 
		image_dir=test_image_dir, 
		ishape=ishape, 
		abox_2dtensor=abox_2dtensor, 
		iou_thresholds=iou_thresholds, 
		total_examples=total_test_examples,
		total_classes=total_classes, 
		anchor_sampling=anchor_sampling)

	loss = np.zeros(total_test_examples)
	total_bboxes = 0
	total_pboxes = 0
	TP = 0
	FN = 0
	FP = 0

	for batch in range(total_test_examples):
		batchx_4dtensor, batchy_2dtensor, bboxes = next(gen)
		batch_loss = model.test_on_batch(batchx_4dtensor, batchy_2dtensor)
		loss[batch] = batch_loss

		prediction = model.predict_on_batch(batchx_4dtensor) # (h*w*k, total_classes+1+4)
		boxclz_2dtensor, valid_outputs = nsm(
			abox_2dtensor=abox_2dtensor, 
			prediction=prediction, 
			nsm_iou_threshold=nsm_iou_threshold,
			nsm_score_threshold=nsm_score_threshold,
			nsm_max_output_size=nsm_max_output_size,
			total_classes=total_classes)
		boxclz_2dtensor = boxclz_2dtensor[:valid_outputs]
		pboxes = list(boxclz_2dtensor.numpy())

		tp1 = 0
		tp2 = 0
		fn = 0
		fp = 0
		
		for i in range(len(bboxes)):
			intersected = 0
			for j in range(len(pboxes)):
				iou = comiou(bbox=bboxes[i], pred_bbox=pboxes[j])
				if iou >= 0.5:
					intersected = 1
					break

			tp1 += intersected
			fn += int(not intersected)

		for i in range(len(pboxes)):
			intersected = 0
			for j in range(len(bboxes)):
				iou = comiou(bbox=bboxes[j], pred_bbox=pboxes[i])
				if iou >= 0.5:
					intersected = 1
					break

			tp2 += intersected
			fp += int(not intersected)

		total_bboxes += len(bboxes)
		total_pboxes += len(pboxes)
		TP += min(tp1, tp2)
		FN += fn
		FP += fp

		print('-', end='')
		if batch%100==99:
			print('{:.2f}%'.format((batch+1)*100/total_test_examples), end='\n')

	precision = TP/(TP + FP + 0.0001)
	recall = TP/(TP + FN + 0.0001)

	mean_loss = float(np.mean(loss, axis=-1))
	if mean_loss < min_loss:
		min_loss = mean_loss
		model.save_weights('{}/weights.h5'.format(output_path))

	if precision > max_precision:
		max_precision = precision
		model.save_weights('{}/weights_best_precision.h5'.format(output_path))

	if recall > max_recall:
		max_recall = recall
		model.save_weights('{}/weights_best_recall.h5'.format(output_path))

	if precision + recall > max_precision_recall:
		max_precision_recall = precision + recall
		model.save_weights('{}/weights_best_precision_recall.h5'.format(output_path))

	print('\nLoss: {:.3f}/{:.3f}, Total bboxes: {}, Total pboxes: {}, TP: {}, FN: {}, FP: {}, Precision: {:.3f}/{:.3f}, Recall: {:.3f}/{:.3f}, Precision & Recall: {:.3f}/{:.3f}'.format(mean_loss, min_loss, total_bboxes, total_pboxes, TP, FN, FP, precision, max_precision, recall, max_recall, precision + recall, max_precision_recall))

