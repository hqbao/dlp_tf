import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from vggface2_datagen import genid
from model import build_rec_model


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

img_dir_path = '../datasets/vggface2/train_refined_resized/'
test_anno_file_path = 'anno/vggface2_test2_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[16, 16, 64], [2, [2, 2]], [2, [2, 2]], [2, [2, 2]]]
dense_settings = []
total_anchor_identities = 1000
total_same_anchors = 20
total_identities = 100
total_same = 10

model = build_rec_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings)
# model.summary()
model.load_weights('{}/clz/weights1.h5'.format(output_path), by_name=True)

E1 = np.load('{}/embedding3d.npy'.format(output_path))
Y1 = np.load('{}/batchy1d.npy'.format(output_path))

batchx4d, Y2 = genid(
	anno_file_path=test_anno_file_path,
	img_dir_path=img_dir_path,
	ishape=ishape,
	total_identities=total_identities,
	total_same_identities=total_same)

print('Start: {}'.format(datetime.now().time()))
E2 = model.predict_on_batch(batchx4d) # (total_identities*total_same, embedding_dims)
print('End: {}'.format(datetime.now().time()))
E2 = tf.reshape(tensor=E2, shape=[total_identities, total_same, -1]) # (total_identities, total_same, embedding_dims)
E2 = tf.math.reduce_mean(input_tensor=E2, axis=1) # (total_identities, embedding_dims)

print('Start: {}'.format(datetime.now().time()))
M = np.zeros((total_anchor_identities, total_identities), dtype='float32')

for i in range(total_anchor_identities):
	e1 = E1[i]
	for j in range(total_identities):
		e2 = E2[j]
		d = tf.norm(tensor=e1-e2, axis=-1)
		M[i, j] = d
print('End: {}'.format(datetime.now().time()))

M = M.reshape((total_anchor_identities, total_identities))
M = np.argmin(M, axis=0) # (total_identities,)

Y2 = Y2.reshape((total_identities, total_same))
Y2 = Y2[:, 0] # (total_identities,)

c = 0
for i in range(len(M)):
	m = M[i]
	if Y1[m] == Y2[i]:
		c += 1

print('Precision: {}'.format(round(c*100/total_identities, 4)))










