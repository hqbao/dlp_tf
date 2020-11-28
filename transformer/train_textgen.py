import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from datagen import TextgenDatagen
from model import textgen_transformer

text_file = '../datasets/text_generation/deeplearning.txt'
output_path = 'output'

total_seqs = 40
seq_len = 100
batch_size = 10
total_batches = total_seqs//batch_size
epochs = 1000

datagen = TextgenDatagen()
words, total_vocabs, _, _, _ = datagen.load_data(text_file)
print(len(words))

seq2d = np.zeros((total_seqs, seq_len+1), dtype='int32')
for i in range(total_seqs):
	seq = datagen.encode(words[i*seq_len:i*seq_len+seq_len+1])
	seq2d[i, :] = seq

model = textgen_transformer(total_vocabs=total_vocabs, seq_len=seq_len)
# model.summary()
# model.load_weights(output_path+'/weights_.h5')

for epoch in range(epochs):
	np.random.shuffle(seq2d)
	loss = np.zeros(total_batches)
	for i in range(total_batches):
		batchx = seq2d[i*batch_size:i*batch_size+batch_size, :-1]
		batchy = seq2d[i*batch_size:i*batch_size+batch_size, 1:]
		batchy = tf.reshape(tensor=batchy, shape=[-1])
		batchy = tf.one_hot(indices=batchy, depth=total_vocabs)

		batch_loss = model.train_on_batch(batchx, batchy)
		loss[i] = batch_loss

		print('-', end='')

	print('\nLoss: {:.3f}'.format(np.mean(loss)))
	model.save_weights(output_path+'/weights_.h5')



