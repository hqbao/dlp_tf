import numpy as np
from random import shuffle


anno_file_path = 'anno/celeba_id_landmark_anno.txt'
train_anno_file_path = 'anno/celeba_train_landmark_anno.txt'
test_anno_file_path = 'anno/celeba_test_landmark_anno.txt'

anno_file = open(anno_file_path, 'r')
train1_anno_file = open(train_anno_file_path, 'w')
test1_anno_file = open(test_anno_file_path, 'w')
lines = anno_file.readlines()

ishape = [160, 160, 3]
total_identities = 1000
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

	y1, x1, y2, x2 = bbox
	h = y2 - y1
	w = x2 - x1

	y1 -= (ishape[0]-h)//2
	if y1 < 0:
		y1 = 0

	y2 = y1+ishape[0]
	if y2 > 218:
		y2 = 218
		y1 = y2-ishape[0]

	x1 -= (ishape[1]-w)//2
	if x1 < 0:
		x1 = 0

	x2 = x1+ishape[1]
	if x2 > 178:
		x2 = 178
		x1 = x2-ishape[1]

	bbox = [y1, x1, y2, x2]

	A[0] -= bbox[0]
	A[1] -= bbox[1]
	B[0] -= bbox[0]
	B[1] -= bbox[1]
	C[0] -= bbox[0]
	C[1] -= bbox[1]
	D[0] -= bbox[0]
	D[1] -= bbox[1]
	E[0] -= bbox[0]
	E[1] -= bbox[1]
	
	if identity not in id_dist:
		id_dist[identity] = [[image_id]+bbox+A+B+C+D+E]
	else:
		id_dist[identity].append([image_id]+bbox+A+B+C+D+E)

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
test1_yx_list = test_yx_list[:total_identities]

train1xy2d = np.zeros((total_identities*25, 16), dtype='int64')
test1xy2d = np.zeros((total_identities*3, 16), dtype='int64')

for i in range(total_identities):
	for j in range(25):
		image_id = train1_yx_list[i][j][0]
		identity = i
		bbox = train1_yx_list[i][j][1:]
		train1xy2d[i*25+j] = [image_id, identity] + bbox

for i in range(total_identities):
	for j in range(3):
		image_id = test1_yx_list[i][j][0]
		identity = i
		bbox = test1_yx_list[i][j][1:]
		test1xy2d[i*3+j] = [image_id, identity] + bbox

np.random.shuffle(train1xy2d)
np.random.shuffle(test1xy2d)

for i in range(train1xy2d.shape[0]):
	line = str(train1xy2d[i, 0]).zfill(6) + '.jpg ' + ' '.join(list(map(str, list(train1xy2d[i, 1:])))) + '\n'
	train1_anno_file.write(line)

for i in range(test1xy2d.shape[0]):
	line = str(test1xy2d[i, 0]).zfill(6) + '.jpg ' + ' '.join(list(map(str, list(test1xy2d[i, 1:])))) + '\n'
	test1_anno_file.write(line)

print('Train samples: {}, test samples: {}'.format(train1xy2d.shape[0], test1xy2d.shape[0]))

anno_file.close()
train1_anno_file.close()
test1_anno_file.close()


