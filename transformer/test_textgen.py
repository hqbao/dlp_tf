import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datagen import TextgenDatagen
from model import textgen_transformer


text_file = '../datasets/text_generation/deeplearning.txt'
output_path = 'output'

seq_len = 40
performance = 10

datagen = TextgenDatagen()
words, total_vocabs, _, _, _ = datagen.load_data(text_file)
print(len(words), total_vocabs)

model = textgen_transformer(total_vocabs=total_vocabs, seq_len=seq_len)
model.load_weights(output_path+'/weights_.h5')

output = np.zeros((1, seq_len))
output[0, 0:1] = datagen.encode(['deep'])

for i in range(1000):
	prediction = model.predict_on_batch(output)
	idx = i if i < seq_len else seq_len-1
	pred_tokens = [j for j in range(len(prediction[idx])) if prediction[idx, j] > performance/total_vocabs]
	# print(pred_tokens)
	# print(np.array([prediction[idx, j] for j in range(len(prediction[idx])) if prediction[idx, j] > performance/total_vocabs]))
	if len(pred_tokens) < 2:
		pred_token = np.argmax(prediction[idx], axis=-1)
	else:
		pred_token = pred_tokens[np.random.randint(0, len(pred_tokens)-1)]

	if i >= seq_len-1:
		output[0, :-1] = output[0, 1:]
		output[0, -1] = pred_token
	else:
		output[0, i+1] = pred_token

	print(datagen.decode([pred_token])[0], end=' ')



