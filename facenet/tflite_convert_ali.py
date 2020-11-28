import tensorflow as tf
from model import build_ali_model


output_path = 'output'
ishape = [112, 112, 1]

model = build_ali_model(ishape=ishape, mode='test')
# model.summary()
model.load_weights('{}/ali/weights.h5'.format(output_path), by_name=True)
model.save('{}/ali/model'.format(output_path))

converter = tf.lite.TFLiteConverter.from_saved_model('{}/ali/model'.format(output_path))
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_types = [tf.uint8]
# converter.inference_input_type = tf.uint8  # or tf.uint8
# converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_model = converter.convert()
open('{}/ali/ali_model.tflite'.format(output_path), 'wb').write(tflite_model)




