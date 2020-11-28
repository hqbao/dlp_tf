import tensorflow as tf
import numpy as np
from models import build_model as build_model
from datagen import genxy
from utils import genanchors, nsm, comiou


train_anno_file_path = '../datasets/quizmarker/qm/train_anno.txt'
train_image_dir = '../datasets/quizmarker/qm/full_images'
test_anno_file_path = '../datasets/quizmarker/qm/test_anno.txt'
test_image_dir = '../datasets/quizmarker/qm/full_images'
ishape = [240, 200, 3]
ssize = [60, 50]
asizes = [[8, 8]]
resnet_settings = [[5, 5, 20], [2, [1, 1]], [8, [2, 2]]]
total_classes = 2
total_epoches = 1000
output_path = 'output'
iou_thresholds = [0.3, 0.35]
anchor_sampling = 1024
nsm_iou_threshold = 0.1
nsm_score_threshold = 0.9
nsm_max_output_size = 250
total_train_examples = 800
total_test_examples = 170

abox4d = genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes)
abox_4dtensor = tf.constant(value=abox4d, dtype='float32')
abox_2dtensor = tf.reshape(tensor=abox_4dtensor, shape=[-1, 4])

model = build_model(ishape=ishape, resnet_settings=resnet_settings, k=len(asizes), total_classes=total_classes)
# model.summary()
# model.load_weights('{}/weights.h5'.format(output_path), by_name=True)

min_loss = 2**32
max_precision = 0
max_recall = 0

for epoch in range(total_epoches):
	gen = genxy(
		anno_file_path=train_anno_file_path, 
		image_dir=train_image_dir, 
		ishape=ishape, 
		abox_2dtensor=abox_2dtensor, 
		iou_thresholds=iou_thresholds, 
		total_classes=total_classes, 
		anchor_sampling=anchor_sampling,
		mode='train')

	print('\nTrain epoch {}'.format(epoch))
	loss = np.zeros(total_train_examples)

	for batch in range(total_train_examples):
		batchx_4dtensor, batchy_2dtensor, _, _ = next(gen)
		batch_loss = model.train_on_batch(batchx_4dtensor, batchy_2dtensor)
		loss[batch] = batch_loss

		if batch%10==9:
			print('-', end='')
		if batch%1000==999:
			print('{:.3f}%'.format((batch+1)*100/total_train_examples), end='\n')

	mean_loss = float(np.mean(loss, axis=-1))
	print('\nLoss: {:.3f}'.format(mean_loss))
	model.save_weights('{}/_weights.h5'.format(output_path))

	print('\nValidate')

	gen = genxy(
		anno_file_path=test_anno_file_path, 
		image_dir=test_image_dir, 
		ishape=ishape, 
		abox_2dtensor=abox_2dtensor, 
		iou_thresholds=iou_thresholds, 
		total_classes=total_classes, 
		anchor_sampling=anchor_sampling,
		mode='test')

	loss = np.zeros(total_test_examples)
	precision = np.zeros(total_test_examples)
	recall = np.zeros(total_test_examples)
	total_faces = 0
	total_pred_faces = 0
	TP = 0
	FP = 0
	FN = 0

	for batch in range(total_test_examples):
		batchx_4dtensor, batchy_2dtensor, bboxes, _ = next(gen)
		batch_loss = model.test_on_batch(batchx_4dtensor, batchy_2dtensor)
		loss[batch] = batch_loss

		prediction = model.predict_on_batch(batchx_4dtensor) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3, total_classes+1+4)
		boxclz_2dtensor, valid_outputs = nsm(
			abox_2dtensor=abox_2dtensor, 
			prediction=prediction, 
			nsm_iou_threshold=nsm_iou_threshold,
			nsm_score_threshold=nsm_score_threshold,
			nsm_max_output_size=nsm_max_output_size,
			total_classes=total_classes)
		boxclz_2dtensor = boxclz_2dtensor[:valid_outputs]
		pred_bboxes = list(boxclz_2dtensor.numpy())

		total_bboxes = len(bboxes)
		total_pred_bboxes = len(pred_bboxes)
		true_positives = 0

		for i in range(total_bboxes):
			for j in range(total_pred_bboxes):
				iou = comiou(bbox=bboxes[i], pred_bbox=pred_bboxes[j])
				if iou >= iou_thresholds[1]:
					true_positives += 1
					break

		false_negatives = total_bboxes - true_positives
		false_positives = total_pred_bboxes - true_positives
		precision[batch] = true_positives / (true_positives + false_positives + 0.00001)
		recall[batch] = true_positives / (true_positives + false_negatives + 0.00001)

		total_faces += total_bboxes
		total_pred_faces += total_pred_bboxes
		TP += true_positives 
		FP += false_positives
		FN += false_negatives

		if batch%10==9:
			print('-', end='')
		if batch%1000==999:
			print('{:.2f}%'.format((batch+1)*100/total_test_examples), end='\n')

	mean_loss = float(np.mean(loss, axis=-1))
	mean_precision = float(np.mean(precision, axis=-1))
	mean_recall = float(np.mean(recall, axis=-1))

	if mean_loss < min_loss:
		min_loss = mean_loss
		model.save_weights('{}/weights.h5'.format(output_path))

	if mean_precision > max_precision:
		max_precision = mean_precision
		model.save_weights('{}/weights_best_precision.h5'.format(output_path))

	if mean_recall > max_recall:
		max_recall = mean_recall
		model.save_weights('{}/weights_best_recall.h5'.format(output_path))

	print('\nLoss: {:.3f}/{:.3f}, Precision: {:.3f}/{:.3f}, Recall: {:.3f}/{:.3f}, Total bboxes: {}, Total predicted bboxes: {}, TP: {}, FP: {}, FN: {}'.format(mean_loss, min_loss, mean_precision, max_precision, mean_recall, max_recall, total_faces, total_pred_faces, TP, FP, FN))










