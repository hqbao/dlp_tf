import tensorflow as tf
import numpy as np


def positional_encoding(position, d_model):
	def get_angles(pos, i, d_model):
		angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
		return pos * angle_rates

	angle_rads = get_angles(pos=np.arange(position)[:, np.newaxis], i=np.arange(d_model)[np.newaxis, :], d_model=d_model)
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	pos_encoding = angle_rads[np.newaxis, ...]

	return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(x):
	mask = tf.cast(tf.math.equal(x, 0), tf.float32)
	return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
	seq_len = x.shape[1]
	look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
	padding_mask = create_padding_mask(x)
	return tf.maximum(look_ahead_mask, padding_mask)

def multihead_attention(q, k, v, total_heads, depth, mask):
	'''
	'''

	assert k.shape[1] == v.shape[1]

	_, seq_len_q, _ = q.shape
	_, seq_len_k, _ = k.shape
	_, seq_len_v, _ = v.shape
	d_model = total_heads*depth

	q = tf.keras.layers.Dense(units=d_model)(q) # (batch_size, seq_len_q, d_model)
	k = tf.keras.layers.Dense(units=d_model)(k) # (batch_size, seq_len_k, d_model)
	v = tf.keras.layers.Dense(units=d_model)(v) # (batch_size, seq_len_v, d_model)

	q = tf.reshape(tensor=q, shape=[-1, seq_len_q, total_heads, depth]) # (batch_size, seq_len_q, total_heads, depth)
	k = tf.reshape(tensor=k, shape=[-1, seq_len_k, total_heads, depth]) # (batch_size, seq_len_k, total_heads, depth)
	v = tf.reshape(tensor=v, shape=[-1, seq_len_v, total_heads, depth]) # (batch_size, seq_len_v, total_heads, depth)

	qk = tf.einsum('bqnd,bknd->bqkn', q, k) # (batch_size, seq_len_q, seq_len_k, total_heads)
	qk = tf.transpose(a=qk, perm=[0, 3, 1, 2]) # (batch_size, total_heads, seq_len_q, seq_len_k)
	qk = qk/tf.math.sqrt(x=tf.cast(x=k.shape[-1], dtype='float32'))

	if mask is not None:
		qk += (mask * -1e9) # (batch_size, total_heads, seq_len_q, seq_len_k)

	qk = tf.nn.softmax(qk, axis=-1)
	qkv = tf.einsum('bnqk,bknd->bnqd', qk, v) # (batch_size, total_heads, seq_len_q, depth)

	tensor = tf.transpose(a=qkv, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, total_heads, depth)
	tensor = tf.reshape(tensor=tensor, shape=[-1, seq_len_q, d_model])  # (batch_size, seq_len_q, d_model)
	tensor = tf.keras.layers.Dense(d_model)(tensor) # (batch_size, seq_len_q, d_model)

	return tensor

def feed_forward(input_tensor, dff, d_model):
	'''
	'''

	tensor = tf.keras.layers.Dense(units=dff, activation='relu')(input_tensor)  # (batch_size, seq_len, dff)
	tensor = tf.keras.layers.Dense(units=d_model)(tensor)  # (batch_size, seq_len, d_model)
	return tensor

def encoder(intput_tensor, num_layers, total_heads, depth, total_vocabs, dff, maximum_position_encoding, dropout_rate, mask_func):
	'''
	Args:
		intput_tensor: (batch_size, seq_len)
	'''

	d_model = total_heads*depth
	seq_len = intput_tensor.shape[1]
	pos_encoding = positional_encoding(position=maximum_position_encoding, d_model=d_model) 

	# Add embedding and position encoding
	tensor = tf.keras.layers.Embedding(input_dim=total_vocabs, output_dim=d_model)(intput_tensor) # (batch_size, seq_len, d_model)
	tensor *= tf.math.sqrt(tf.cast(d_model, 'float32'))
	tensor += pos_encoding[:, :seq_len, :]
	tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

	for i in range(num_layers):
		_tensor = tensor
		tensor = multihead_attention(q=tensor, k=tensor, v=tensor, total_heads=total_heads, depth=depth, mask=mask_func(intput_tensor)) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Add()([_tensor, tensor]) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tensor) # (batch_size, seq_len, d_model)

		_tensor = tensor
		tensor = feed_forward(input_tensor=tensor, dff=dff, d_model=d_model) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Add()([_tensor, tensor]) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tensor) # (batch_size, seq_len, d_model)

	return tensor

def decoder(enc_output_tensor, enc_input_tensor, dec_input_tensor, num_layers, total_heads, depth, total_vocabs, dff, maximum_position_encoding, dropout_rate, mask_func1, mask_func2):
	'''
	'''

	d_model = total_heads*depth
	seq_len = dec_input_tensor.shape[1]
	pos_encoding = positional_encoding(position=maximum_position_encoding, d_model=d_model) 

	# Add embedding and position encoding
	tensor = tf.keras.layers.Embedding(input_dim=total_vocabs, output_dim=d_model)(dec_input_tensor) # (batch_size, seq_len, d_model)
	tensor *= tf.math.sqrt(tf.cast(d_model, 'float32'))
	tensor += pos_encoding[:, :seq_len, :]
	tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

	for i in range(num_layers):
		_tensor = tensor
		tensor = multihead_attention(q=tensor, k=tensor, v=tensor, total_heads=total_heads, depth=depth, mask=mask_func1(dec_input_tensor)) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Add()([_tensor, tensor]) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tensor) # (batch_size, seq_len, d_model)	

		_tensor = tensor
		tensor = multihead_attention(q=tensor, k=enc_output_tensor, v=enc_output_tensor, total_heads=total_heads, depth=depth, mask=mask_func2(enc_input_tensor)) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Add()([_tensor, tensor]) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tensor) # (batch_size, seq_len, d_model)		

		_tensor = tensor
		tensor = feed_forward(input_tensor=tensor, dff=dff, d_model=d_model) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.Add()([_tensor, tensor]) # (batch_size, seq_len, d_model)
		tensor = tf.keras.layers.LayerNormalization(epsilon=1e-6)(tensor) # (batch_size, seq_len, d_model)
		
	return tensor

def transformer_loss(total_vocabs, pad_token):
	def loss(y_true, y_pred):
		'''
		Args:
			y_true: (batch_size*seq_len, total_vocabs)
			y_pred: (batch_size*seq_len, total_vocabs)
		'''
		
		y_true = tf.reshape(tensor=y_true, shape=[-1, total_vocabs])
		y_pred = tf.reshape(tensor=y_pred, shape=[-1, total_vocabs])

		# Not calculate loss for pad tokens
		# max_index_1dtensor = tf.math.argmax(input=y_true, axis=-1) # (batch_size*seq_len,)
		# mask = tf.where(condition=tf.math.equal(x=max_index_1dtensor, y=pad_token), x=0.0, y=1.0) # (batch_size*seq_len,)

		loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred) # (batch_size*seq_len,)
		# loss = loss*mask
		loss = tf.math.reduce_mean(input_tensor=loss, axis=-1)

		return loss

	return loss

def mapping_transformer(total_vocabs, enc_seq_len, dec_seq_len, pad_token):
	'''
	'''
	
	enc_num_layers = 3
	dec_num_layers = 3
	total_heads = 2
	depth = 64
	dff = 256
	maximum_position_encoding=10000

	enc_intput_tensor = tf.keras.Input(shape=[enc_seq_len], batch_size=None)
	dec_intput_tensor = tf.keras.Input(shape=[dec_seq_len], batch_size=None)

	tensor = encoder(
		intput_tensor=enc_intput_tensor, 
		num_layers=enc_num_layers, 
		total_heads=total_heads, 
		depth=depth, 
		total_vocabs=total_vocabs, 
		dff=dff, 
		maximum_position_encoding=maximum_position_encoding, 
		dropout_rate=0.2,
		mask_func=create_padding_mask)

	tensor = decoder(
		enc_output_tensor=tensor,
		enc_input_tensor=enc_intput_tensor,
		dec_input_tensor=dec_intput_tensor, 
		num_layers=dec_num_layers, 
		total_heads=total_heads, 
		depth=depth, 
		total_vocabs=total_vocabs, 
		dff=dff, 
		maximum_position_encoding=maximum_position_encoding, 
		dropout_rate=0.2,
		mask_func1=create_look_ahead_mask,
		mask_func2=create_padding_mask)

	tensor = tf.keras.layers.Dense(units=total_vocabs)(tensor)
	tensor = tf.keras.layers.Activation(activation='softmax')(tensor)
	tensor = tf.reshape(tensor=tensor, shape=[-1, total_vocabs])

	model = tf.keras.models.Model(inputs=[enc_intput_tensor, dec_intput_tensor], outputs=tensor)
	model.compile(optimizer=tf.keras.optimizers.Adam(), loss=transformer_loss(total_vocabs, pad_token))

	return model

def textgen_transformer(total_vocabs, seq_len):
	'''
	'''
	
	num_layers = 3
	total_heads = 2
	depth = 64
	dff = 256
	maximum_position_encoding=10000

	intput_tensor = tf.keras.Input(shape=[seq_len], batch_size=None)

	tensor = encoder(
		intput_tensor=intput_tensor, 
		num_layers=num_layers, 
		total_heads=total_heads, 
		depth=depth, 
		total_vocabs=total_vocabs, 
		dff=dff, 
		maximum_position_encoding=maximum_position_encoding, 
		dropout_rate=0.2,
		mask_func=create_look_ahead_mask)

	tensor = tf.keras.layers.Dense(units=total_vocabs)(tensor)
	tensor = tf.keras.layers.Activation(activation='softmax')(tensor)
	tensor = tf.reshape(tensor=tensor, shape=[-1, total_vocabs])

	model = tf.keras.models.Model(inputs=intput_tensor, outputs=tensor)
	model.compile(optimizer=tf.keras.optimizers.Adam(), loss=transformer_loss(total_vocabs, 0))

	return model

