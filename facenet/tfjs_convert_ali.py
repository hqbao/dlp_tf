import tensorflow as tf
from model import build_ali_model


output_path = 'output'
ishape = [112, 112, 1]

model = build_ali_model(ishape=ishape, mode='test')
# model.summary()
model.load_weights('{}/ali/weights.h5'.format(output_path), by_name=True)
model.save('{}/ali/model'.format(output_path))

