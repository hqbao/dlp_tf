import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from vggface2_datagen import genid
from model import build_rec_model


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

img_dir_path = '../datasets/vggface2/train_refined_resized/'
anno_file_path = 'anno/vggface2_train2_anno.txt'
output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[16, 16, 64], [2, [2, 2]], [2, [2, 2]], [2, [2, 2]]]
dense_settings = []
total_identities = 1000
total_same = 20

model = build_rec_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings)
model.summary()
model.load_weights('{}/clz/weights1.h5'.format(output_path), by_name=True)

batchx4d, batchy1d = genid(
	anno_file_path=anno_file_path,
	img_dir_path=img_dir_path,
	ishape=ishape,
	total_identities=total_identities,
	total_same_identities=total_same)

print('Start: {}'.format(datetime.now().time()))
prediction = model.predict_on_batch(batchx4d) # (total_identities*total_same, embedding_dims)
print('End: {}'.format(datetime.now().time()))

prediction = tf.reshape(tensor=prediction, shape=[total_identities, total_same, -1]).numpy() 
prediction = tf.math.reduce_mean(input_tensor=prediction, axis=1) # (total_identities, embedding_dims)
np.save('{}/embedding3d.npy'.format(output_path), prediction)
batchy1d = tf.reshape(tensor=batchy1d, shape=[total_identities, total_same]).numpy()
batchy1d = batchy1d[:, 0] # (total_identities, embedding_dims)
np.save('{}/batchy1d.npy'.format(output_path), batchy1d)









