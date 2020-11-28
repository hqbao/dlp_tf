import numpy as np
from random import shuffle


anno_file_path = 'anno/celeba_id_landmark_anno.txt'
train1_anno_file_path = 'anno/celeba_train1_anno.txt'
train2_anno_file_path = 'anno/celeba_train2_anno.txt'
test1_anno_file_path = 'anno/celeba_test1_anno.txt'
test2_anno_file_path = 'anno/celeba_test2_anno.txt'
total_identities = 1000

anno_file = open(anno_file_path, 'r')
train1_anno_file = open(train1_anno_file_path, 'w')
train2_anno_file = open(train2_anno_file_path, 'w')
test1_anno_file = open(test1_anno_file_path, 'w')
test2_anno_file = open(test2_anno_file_path, 'w')
lines = anno_file.readlines()

id_dist = {}

for line_idx in range(len(lines)):
	line = lines[line_idx][:-1]
	anno = line.split(' ')
	image_id = int(anno[0][:-4])
	identity = int(anno[1])
	landmark = list(map(int, anno[2:]))

	A = [landmark[1], landmark[0]]
	B = [landmark[3], landmark[2]]
	C = [landmark[7], landmark[6]]
	D = [landmark[9], landmark[8]]
	E = [landmark[5], landmark[4]]

	x_ab = B[1] - A[1]
	y_ac = D[0] - A[0]
	x_ea = E[1] - A[1]
	x_eb = B[1] - E[1]

	if x_ea <= 0 or x_eb <= 0:
		continue

	if max(x_ea/x_eb, x_eb/x_ea) > 10:
		continue

	x_ea_per_eb = abs(min(x_ea/x_eb, 2))
	x_eb_per_ea = abs(min(x_eb/x_ea, 2))

	left = int(A[1] - 0.5*x_ab - (0.4*x_ea_per_eb)*x_ab)
	right = int(B[1] + 0.5*x_ab + (0.4*x_eb_per_ea)*x_ab)
	top = int(A[0] - y_ac)
	bottom = int(top + 1.1*(right - left))

	bbox = [top, left, bottom, right] # [y1, x1, y2, x2]
	
	if identity not in id_dist:
		id_dist[identity] = [[image_id]+bbox]
	else:
		id_dist[identity].append([image_id]+bbox)

yx = []

for identity in sorted(id_dist):
	yx.append(id_dist[identity])

train_yx_list = []
test_yx_list = []

for i in range(len(yx)):
	x_list = yx[i]
	x_list_len = len(x_list)

	if x_list_len >= 28:
		train_yx_list.append(x_list[:25])
		test_yx_list.append(x_list[25:28])

print('Train identities: {}'.format(len(train_yx_list)))
print('Test identities: {}'.format(len(test_yx_list)))

train1_yx_list = train_yx_list[:total_identities]
train2_yx_list = train_yx_list[total_identities:2*total_identities]
test1_yx_list = test_yx_list[:total_identities]
test2_yx_list = test_yx_list[total_identities:2*total_identities]

train1xy2d = np.zeros((total_identities*25, 6), dtype='int64')
train2xy2d = np.zeros((total_identities*25, 6), dtype='int64')
test1xy2d = np.zeros((total_identities*3, 6), dtype='int64')
test2xy2d = np.zeros((total_identities*3, 6), dtype='int64')

for i in range(total_identities):
	for j in range(25):
		image_id = train1_yx_list[i][j][0]
		identity = i
		bbox = train1_yx_list[i][j][1:]
		train1xy2d[i*25+j] = [image_id, identity] + bbox

for i in range(total_identities):
	for j in range(25):
		image_id = train2_yx_list[i][j][0]
		identity = i
		bbox = train2_yx_list[i][j][1:]
		train2xy2d[i*25+j] = [image_id, identity] + bbox

for i in range(total_identities):
	for j in range(3):
		image_id = test1_yx_list[i][j][0]
		identity = i
		bbox = test1_yx_list[i][j][1:]
		test1xy2d[i*3+j] = [image_id, identity] + bbox

for i in range(total_identities):
	for j in range(3):
		image_id = test2_yx_list[i][j][0]
		identity = i
		bbox = test2_yx_list[i][j][1:]
		test2xy2d[i*3+j] = [image_id, identity] + bbox

np.random.shuffle(train1xy2d)
np.random.shuffle(train2xy2d)
np.random.shuffle(test1xy2d)
np.random.shuffle(test2xy2d)

for i in range(train1xy2d.shape[0]):
	line = str(train1xy2d[i, 0]).zfill(6) + '.jpg ' + ' '.join(list(map(str, list(train1xy2d[i, 1:])))) + '\n'
	train1_anno_file.write(line)

for i in range(train2xy2d.shape[0]):
	line = str(train2xy2d[i, 0]).zfill(6) + '.jpg ' + ' '.join(list(map(str, list(train2xy2d[i, 1:])))) + '\n'
	train2_anno_file.write(line)

for i in range(test1xy2d.shape[0]):
	line = str(test1xy2d[i, 0]).zfill(6) + '.jpg ' + ' '.join(list(map(str, list(test1xy2d[i, 1:])))) + '\n'
	test1_anno_file.write(line)

for i in range(test2xy2d.shape[0]):
	line = str(test2xy2d[i, 0]).zfill(6) + '.jpg ' + ' '.join(list(map(str, list(test2xy2d[i, 1:])))) + '\n'
	test2_anno_file.write(line)

print('Train samples: {}, {}, test samples: {}, {}'.format(train1xy2d.shape[0], train2xy2d.shape[0], test1xy2d.shape[0], test2xy2d.shape[0]))

anno_file.close()
train1_anno_file.close()
test1_anno_file.close()


