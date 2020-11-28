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
total_identities = 100
total_same_identities = 20

model = build_rec_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings)
# model.summary()
model.load_weights('{}/clz/weights1.h5'.format(output_path), by_name=True)

batchx4d = genid(
	anno_file_path=test_anno_file_path,
	img_dir_path=img_dir_path,
	ishape=ishape,
	total_identities=total_identities,
	total_same_identities=total_same_identities)

print('Start: {}'.format(datetime.now().time()))
pred_batchy_2dtensor = model.predict_on_batch(batchx4d) # (total_identities*total_same_identities, embedding_dims)
print('End: {}'.format(datetime.now().time()))

pred_batchy_3dtensor = tf.reshape(tensor=pred_batchy_2dtensor, shape=[total_identities, total_same_identities, -1])
pred_batchy_2dtensor = tf.math.reduce_mean(input_tensor=pred_batchy_3dtensor, axis=1) # (total_identities, embedding_dims)
Y = pred_batchy_2dtensor
X = pred_batchy_2dtensor

print('Start: {}'.format(datetime.now().time()))
V = np.zeros((total_identities, total_identities), dtype='float32')
for i in range(total_identities):
	y = Y[i] # (embedding_dims,)
	for j in range(total_identities):
		x = X[j] # (embedding_dims,)
		d = tf.norm(tensor=y-x, axis=-1)
		V[j, i] = d
print('End: {}'.format(datetime.now().time()))

plt.figure(figsize=(7.35, 7.35))
plt.imshow(V)
plt.show()






