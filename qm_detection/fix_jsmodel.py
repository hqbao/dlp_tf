import json
from pprint import pprint

modelJsonStr = open('output/tfjs/model.json')
modelJson = json.load(modelJsonStr)
layers = modelJson['modelTopology']['model_config']['config']['layers']

for i in range(len(layers)):
	if layers[i]['class_name'] == 'TensorFlowOpLayer':
		if 'Cast' in layers[i]['name']:
			modelJson['modelTopology']['model_config']['config']['layers'][i]['class_name'] = 'tf_op_layer_cast'

for i in range(len(layers)):
	pprint(layers[i]['class_name'])

with open('output/tfjs/'+'_model.json', 'w') as f:
    json.dump(modelJson, f)