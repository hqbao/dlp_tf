import tensorflow as tf
from pprint import pprint


output_path = 'output'
ishape = [112, 112, 1]

interpreter = tf.lite.Interpreter(model_path='{}/clz/rec_model.tflite'.format(output_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
pprint(input_details)
pprint(output_details)

# batch_x = tf.expand_dims(input=x, axis=0)
# interpreter.set_tensor(tensor_index=input_details[0]['index'], value=batch_x)
# interpreter.invoke()
# prediction = interpreter.get_tensor(output_details[0]['index'])





