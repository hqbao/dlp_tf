import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from datetime import datetime
from model import mapping_transformer
from datagen import MappingDatagen

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

text_file = '../datasets/chat_bot/cornell-movie-dialogs-corpus/seq_sentences.txt'
output_path = 'output'

datagen = MappingDatagen()
sentences, total_vocabs, PAD_TOKEN, START_TOKEN, END_TOKEN = datagen.load_data(text_file)
print(total_vocabs, PAD_TOKEN, START_TOKEN, END_TOKEN)

# sentence_lens = [len(sentence) for sentence in sentences]
# plt.plot(sentence_lens)
# plt.show()

total_examples = 10001
seq_len = 50
batch_size = 100
total_batches = total_examples//batch_size
epochs = 1000

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

model = mapping_transformer(total_vocabs=total_vocabs, enc_seq_len=seq_len, dec_seq_len=seq_len-1, pad_token=PAD_TOKEN)
# model.load_weights(output_path+'/weights_.h5')

for epoch in range(epochs):
	loss = np.zeros(total_batches)
	for i in range(total_batches):
		batch_enc = sentence2d[i*batch_size:i*batch_size+batch_size]
		batch_enc_input = batch_enc[:, :]

		batch_dec = sentence2d[i*batch_size+1:i*batch_size+batch_size+1]
		batch_dec_input = batch_dec[:, :-1]
		batch_dec_output = batch_dec[:, 1:]
		batch_dec_output = tf.reshape(tensor=batch_dec_output, shape=[-1])
		batch_dec_output = tf.one_hot(indices=batch_dec_output, depth=total_vocabs)

		batch_loss = model.train_on_batch([batch_enc_input, batch_dec_input], batch_dec_output)
		loss[i] = batch_loss

		print('-', end='')

	print('\nLoss: {:.3f}'.format(np.mean(loss)))
	model.save_weights(output_path+'/weights_.h5')

