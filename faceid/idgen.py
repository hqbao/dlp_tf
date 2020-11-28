import tensorflow as tf
import numpy as np
import skimage.io as io
from os import listdir
from pprint import pprint
from utils import generate_embedding


output_path = 'outputs'
tflite_model_path = 'tflite_models'
ids_dir = 'ids'
ishape = [112, 112, 3]
embedding_dims = 256
id_samples = 9

interpreter = tf.lite.Interpreter(model_path=tflite_model_path+'/rec_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# pprint(input_details)
# pprint(output_details)

IDs = listdir(ids_dir)
while '.DS_Store' in IDs: IDs.remove('.DS_Store')
total_ids = len(IDs)
print(IDs)

embedding2d = np.zeros((total_ids, embedding_dims))

for i in range(total_ids):
	ID = IDs[i]
	file_names = listdir(ids_dir+'/'+ID)
	while '.DS_Store' in file_names: file_names.remove('.DS_Store')
	embeddings = []
	for j in range(len(file_names)):
		file_name = file_names[j]
		pix = io.imread(ids_dir+'/'+ID+'/'+file_name)
		pix = np.mean(pix, axis=-1, keepdims=True)
		pix = np.concatenate([pix, pix, pix], axis=-1)
		pix = np.array(pix, dtype='int32')

		# Origin
		embedding = generate_embedding(
			interpreter=interpreter, 
			input_details=input_details, 
			output_details=output_details, 
			pix=pix)
		embeddings.append(embedding)

		# Flip
		pix = np.fliplr(pix)
		embedding = generate_embedding(
			interpreter=interpreter, 
			input_details=input_details, 
			output_details=output_details, 
			pix=pix)
		embeddings.append(embedding)

	embedding2d[i] = np.mean(np.array(embeddings, dtype='float32'), axis=0)

np.save(output_path+'/embedding2d.npy', embedding2d)

