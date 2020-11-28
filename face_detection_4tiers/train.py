import tensorflow as tf
import numpy as np
from models import build_model as build_model
from datagen import load_dataset, genxy
from utils import genanchors, nsm, comiou, match


output_path = 'output'
train_anno_file_path = '../datasets/widerface/train.txt'
train_image_dir = '../datasets/widerface/train'
test_anno_file_path = '../datasets/widerface/test.txt'
test_image_dir = '../datasets/widerface/test'
ishape = [1024, 1024, 3]
ssizes = [
	[256, 256],
	[128, 128],
	[64, 64], 
	[32, 32], 
]
asizes = [
	[[64, 64]],
	[[128, 128]],
	[[256, 256]],
	[[512, 512]],
]
total_classes = 1
resnet_settings = [[8, 8, 32], [8, [2, 2]], [8, [2, 2]], [8, [2, 2]], [4, [2, 2]]]
top_down_pyramid_size = 64
total_epoches = 100
iou_thresholds = [[0.5, 0.6], [0.45, 0.55], [0.4, 0.5], [0.35, 0.45]]
nsm_iou_threshold = 0.2
nsm_score_threshold = 0.8
nsm_max_output_size = 100
anchor_samplings = [512, 256, 128, 64]
total_train_examples = 4
total_test_examples = 4

a1box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[0], asizes=asizes[0]), dtype='float32') # (h1*w1*k1, 4)
a2box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[1], asizes=asizes[1]), dtype='float32') # (h2*w2*k2, 4)
a3box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[2], asizes=asizes[2]), dtype='float32') # (h3*w3*k3, 4)
a4box_2dtensor = tf.constant(value=genanchors(isize=ishape[:2], ssize=ssizes[3], asizes=asizes[3]), dtype='float32') # (h4*w4*k4, 4)
abox_2dtensors = [a1box_2dtensor, a2box_2dtensor, a3box_2dtensor, a4box_2dtensor]
abox_2dtensor = tf.concat(values=abox_2dtensors, axis=0)

model = build_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	top_down_pyramid_size=top_down_pyramid_size,
	k=[len(asizes[0]), len(asizes[1]), len(asizes[2]), len(asizes[3])], 
	total_classes=total_classes)
# model.summary()
model.load_weights('{}/weights_.h5'.format(output_path), by_name=True)

min_loss = 2**32

max_precision = 0
max_x_precision = 0
max_s_precision = 0
max_m_precision = 0
max_l_precision = 0

max_recall = 0
max_x_recall = 0
max_s_recall = 0
max_m_recall = 0
max_l_recall = 0

train_dataset = load_dataset(anno_file_path=train_anno_file_path)
test_dataset = load_dataset(anno_file_path=test_anno_file_path)

for epoch in range(total_epoches):
	# tf.keras.backend.set_value(model.optimizer.learning_rate, 0.001)

	gen = genxy(
		dataset=train_dataset, 
		image_dir=train_image_dir, 
		ishape=ishape, 
		abox_2dtensors=abox_2dtensors, 
		iou_thresholds=iou_thresholds, 
		total_examples=total_train_examples,
		total_classes=total_classes, 
		anchor_samplings=anchor_samplings)

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

	gen = genxy(
		dataset=test_dataset, 
		image_dir=test_image_dir, 
		ishape=ishape, 
		abox_2dtensors=abox_2dtensors, 
		iou_thresholds=iou_thresholds, 
		total_examples=total_test_examples,
		total_classes=total_classes, 
		anchor_samplings=anchor_samplings)

	loss = np.zeros(total_test_examples)
	precision = np.zeros(total_test_examples)
	recall = np.zeros(total_test_examples)
	total_x_bboxes = 0
	total_s_bboxes = 0
	total_m_bboxes = 0
	total_l_bboxes = 0
	total_pboxes = 0

	x_TP = 0
	s_TP = 0
	m_TP = 0
	l_TP = 0

	x_FN = 0
	s_FN = 0
	m_FN = 0
	l_FN = 0

	x_FP = 0
	s_FP = 0
	m_FP = 0
	l_FP = 0
	
	TP = 0
	FN = 0
	FP = 0

	for batch in range(total_test_examples):
		batchx_4dtensor, batchy_2dtensor, bboxes = next(gen)
		batch_loss = model.test_on_batch(batchx_4dtensor, batchy_2dtensor)
		loss[batch] = batch_loss

		prediction = model.predict_on_batch(batchx_4dtensor) # (h1*w1*k1 + h2*w2*k2 + h3*w3*k3 + h4*w4*k4, total_classes+1+4)
		boxclz_2dtensor, valid_outputs = nsm(
			abox_2dtensor=abox_2dtensor, 
			prediction=prediction, 
			nsm_iou_threshold=nsm_iou_threshold,
			nsm_score_threshold=nsm_score_threshold,
			nsm_max_output_size=nsm_max_output_size,
			total_classes=total_classes)
		boxclz_2dtensor = boxclz_2dtensor[:valid_outputs]
		pred_bboxes = list(boxclz_2dtensor.numpy())

		total_pboxes += len(pred_bboxes)
		total_x_boxes, total_s_boxes, total_m_boxes, total_l_boxes, x_tp, s_tp, m_tp, l_tp, x_fn, s_fn, m_fn, l_fn, x_fp, s_fp, m_fp, l_fp = match(boxes=bboxes, pboxes=pred_bboxes)

		total_x_bboxes += total_x_boxes
		total_s_bboxes += total_s_boxes
		total_m_bboxes += total_m_boxes
		total_l_bboxes += total_l_boxes

		x_TP += x_tp
		s_TP += s_tp
		m_TP += m_tp
		l_TP += l_tp 

		x_FN += x_fn
		s_FN += s_fn
		m_FN += m_fn
		l_FN += l_fn

		x_FP += x_fp
		s_FP += s_fp
		m_FP += m_fp
		l_FP += l_fp
		
		TP += x_tp+s_tp+m_tp+l_tp
		FN += x_fn+s_fn+m_fn+l_fn
		FP += x_fp+s_fp+m_fp+l_fp

		print('-', end='')
		if batch%100==99:
			print('{:.2f}%'.format((batch+1)*100/total_test_examples), end='\n')

	x_precision = x_TP/(x_TP + x_FP + 0.0001)
	s_precision = s_TP/(s_TP + s_FP + 0.0001)
	m_precision = m_TP/(m_TP + m_FP + 0.0001)
	l_precision = l_TP/(l_TP + l_FP + 0.0001)

	x_recall = x_TP/(x_TP + x_FN + 0.0001)
	s_recall = s_TP/(s_TP + s_FN + 0.0001)
	m_recall = m_TP/(m_TP + m_FN + 0.0001)
	l_recall = l_TP/(l_TP + l_FN + 0.0001)

	precision = TP/(TP + FP + 0.0001)
	recall = TP/(TP + FN + 0.0001)

	mean_loss = float(np.mean(loss, axis=-1))
	if mean_loss < min_loss:
		min_loss = mean_loss
		model.save_weights('{}/weights.h5'.format(output_path))

	if precision > max_precision:
		max_precision = precision
		model.save_weights('{}/weights_best_precision.h5'.format(output_path))

	if x_precision > max_x_precision:
		max_x_precision = x_precision
		model.save_weights('{}/weights_best_x_precision.h5'.format(output_path))

	if s_precision > max_s_precision:
		max_s_precision = s_precision
		model.save_weights('{}/weights_best_s_precision.h5'.format(output_path))

	if m_precision > max_m_precision:
		max_m_precision = m_precision
		model.save_weights('{}/weights_best_m_precision.h5'.format(output_path))

	if l_precision > max_l_precision:
		max_l_precision = l_precision
		model.save_weights('{}/weights_best_l_precision.h5'.format(output_path))

	if recall > max_recall:
		max_recall = recall
		model.save_weights('{}/weights_best_recall.h5'.format(output_path))

	if x_recall > max_x_recall:
		max_x_recall = x_recall
		model.save_weights('{}/weights_best_x_recall.h5'.format(output_path))

	if s_recall > max_s_recall:
		max_s_recall = s_recall
		model.save_weights('{}/weights_best_s_recall.h5'.format(output_path))

	if m_recall > max_m_recall:
		max_m_recall = m_recall
		model.save_weights('{}/weights_best_m_recall.h5'.format(output_path))

	if l_recall > max_l_recall:
		max_l_recall = l_recall
		model.save_weights('{}/weights_best_l_recall.h5'.format(output_path))

	print('\nLoss: {:.3f}/{:.3f}, Total bboxes: {}, Total predicted bboxes: {}, TP: {}, FN: {}, FP: {}, Precision: {:.3f}/{:.3f}, Recall: {:.3f}/{:.3f}'.format(mean_loss, min_loss, total_x_bboxes+total_s_bboxes+total_m_bboxes+total_l_bboxes, total_pboxes, TP, FN, FP, precision, max_precision, recall, max_recall))
	print('xPrecision: {:.3f}/{:.3f}, xRecall: {:.3f}/{:.3f}, xbboxes: {}, xTP: {}, xFN: {}, xFP: {}'.format(x_precision, max_x_precision, x_recall, max_x_recall, total_x_bboxes, x_TP, x_FN, x_FP))
	print('sPrecision: {:.3f}/{:.3f}, sRecall: {:.3f}/{:.3f}, sbboxes: {}, sTP: {}, sFN: {}, sFP: {}'.format(s_precision, max_s_precision, s_recall, max_s_recall, total_s_bboxes, s_TP, s_FN, s_FP))
	print('mPrecision: {:.3f}/{:.3f}, mRecall: {:.3f}/{:.3f}, mbboxes: {}, mTP: {}, mFN: {}, mFP: {}'.format(m_precision, max_m_precision, m_recall, max_m_recall, total_m_bboxes, m_TP, m_FN, m_FP))
	print('lPrecision: {:.3f}/{:.3f}, lRecall: {:.3f}/{:.3f}, lbboxes: {}, lTP: {}, lFN: {}, lFP: {}'.format(l_precision, max_l_precision, l_recall, max_l_recall, total_l_bboxes, l_TP, l_FN, l_FP))










