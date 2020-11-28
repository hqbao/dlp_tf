import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from datetime import datetime
from model import mapping_transformer
from datagen import MappingDatagen

text_file = '../datasets/chat_bot/cornell-movie-dialogs-corpus/seq_sentences.txt'
output_path = 'output'

datagen = MappingDatagen()
sentences, total_vocabs, PAD_TOKEN, START_TOKEN, END_TOKEN = datagen.load_data(text_file)
print(total_vocabs, PAD_TOKEN, START_TOKEN, END_TOKEN)

total_examples = 10001
seq_len = 50
performance = 1000

model = mapping_transformer(total_vocabs=total_vocabs, enc_seq_len=seq_len, dec_seq_len=seq_len-1, pad_token=PAD_TOKEN)
model.load_weights(output_path+'/weights_.h5')

sentence2d = np.zeros((total_examples, seq_len), dtype='int32')
for i in range(total_examples):
	sentence = sentences[i]
	enc_sentence = datagen.encode(sentence)
	if len(enc_sentence) > seq_len-2:
		enc_sentence = enc_sentence[:seq_len-2]

	sentence2d[i, 0] = START_TOKEN
	sentence2d[i, 1:1+len(enc_sentence)] = enc_sentence
	sentence2d[i, 1+len(enc_sentence)] = END_TOKEN
	sentence2d[i, 2+len(enc_sentence):] = PAD_TOKEN

for x in range(10):
	enc_input = sentence2d[x:x+1, :]
	words = datagen.decode(enc_input[0])
	for word in words:
		if word != "_PAD_":
			print(word, end=' ')

	print('\n')

	output = np.zeros((1, seq_len-1))
	output[0, 0] = START_TOKEN
	for i in range(seq_len-2):
		prediction = model.predict_on_batch([enc_input, output])
		idx = i if i < seq_len else seq_len-1
		pred_tokens = [j for j in range(len(prediction[idx])) if prediction[idx, j] > performance/total_vocabs]
		# print(pred_tokens)
		# print(np.array([prediction[idx, j] for j in range(len(prediction[idx])) if prediction[idx, j] > performance/total_vocabs]))
		if len(pred_tokens) < 2:
			pred_token = np.argmax(prediction[idx], axis=-1)
		else:
			pred_token = pred_tokens[np.random.randint(0, len(pred_tokens)-1)]

		print(datagen.decode([pred_token])[0], end=' ')

		if pred_token == END_TOKEN:
			break

		output[0, i+1] = pred_token
		
	print('\n')


# enc_sentence = datagen.encode('hi, how are you?')
# sentence2d = np.zeros((1, seq_len), dtype='int32')
# sentence2d[0, 0] = START_TOKEN
# sentence2d[0, 1:1+len(enc_sentence)] = enc_sentence
# sentence2d[0, 1+len(enc_sentence)] = END_TOKEN
# sentence2d[0, 2+len(enc_sentence):] = PAD_TOKEN

# enc_input = sentence2d
# input_sentence = datagen.decode(enc_input[0])
# print(input_sentence)

# output = np.zeros((1, seq_len-1))
# output[0, 0] = START_TOKEN
# for i in range(seq_len-2):
# 	prediction = model.predict_on_batch([enc_input, output])
# 	idx = i if i < seq_len else seq_len-1
# 	pred_tokens = [j for j in range(len(prediction[idx])) if prediction[idx, j] > performance/total_vocabs]
# 	# print(pred_tokens)
# 	# print(np.array([prediction[idx, j] for j in range(len(prediction[idx])) if prediction[idx, j] > performance/total_vocabs]))
# 	if len(pred_tokens) < 2:
# 		pred_token = np.argmax(prediction[idx], axis=-1)
# 	else:
# 		pred_token = pred_tokens[np.random.randint(0, len(pred_tokens)-1)]

# 	print(datagen.decode([pred_token])[0], end=' ')

# 	if pred_token == END_TOKEN:
# 		break

# 	output[0, i+1] = pred_token
	

