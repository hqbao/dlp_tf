import h5py
from pprint import pprint

filename = 'output/large/weights_best_precision_recall.h5'

with h5py.File(filename, 'r') as f:
    keys = f.keys()
    print(keys)
    layer = f['max_pooling2d']
    # print(layer.keys())
    print(layer)
