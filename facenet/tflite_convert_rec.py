import tensorflow as tf
from model import build_idgen_model


output_path = 'output'
ishape = [112, 112, 3]
resnet_settings = [[16, 16, 64], [2, [2, 2]], [2, [2, 2]], [2, [2, 2]]]
dense_settings = []

model = build_idgen_model(
	ishape=ishape, 
	resnet_settings=resnet_settings, 
	dense_settings=dense_settings)
# model.summary()
model.load_weights('{}/clz/weights1.h5'.format(output_path), by_name=True)
model.save('{}/clz/model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/clz/model'.format(output_path))
# converter.experimental_new_converter = True
tflite_model = converter.convert()
open('{}/clz/rec_model.tflite'.format(output_path), 'wb').write(tflite_model)




