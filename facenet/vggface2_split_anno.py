import cv2
import numpy as np
import os


anno_file_path = 'anno/vggface2_refined_anno.txt'
train1_anno_file_path = 'anno/vggface2_train1_anno.txt'
test1_anno_file_path = 'anno/vggface2_test1_anno.txt'
train2_anno_file_path = 'anno/vggface2_train2_anno.txt'
test2_anno_file_path = 'anno/vggface2_test2_anno.txt'
train3_anno_file_path = 'anno/vggface2_train3_anno.txt'
test3_anno_file_path = 'anno/vggface2_test3_anno.txt'

# total 4407 identities
total_identities_1 = 3000
total_identities_2 = 1200
total_identities_3 = 10

anno_file = open(anno_file_path, 'r')
train1_anno_file = open(train1_anno_file_path, 'w')
test1_anno_file = open(test1_anno_file_path, 'w')
train2_anno_file = open(train2_anno_file_path, 'w')
test2_anno_file = open(test2_anno_file_path, 'w')
train3_anno_file = open(train3_anno_file_path, 'w')
test3_anno_file = open(test3_anno_file_path, 'w')
lines = anno_file.readlines()
total_images = len(lines)

print('Total images: {}'.format(total_images))

yx_dist = {}

for i in range(total_images):
	line = lines[i][:-1]
	anno = line.split(' ')
	id_folder, file_name = anno[0].split('/')
	id = int(anno[1])

	if id not in yx_dist:
		yx_dist[id] = [id_folder + '/' + file_name]
	else:
		yx_dist[id].append(id_folder + '/' + file_name)

yx = []

for identity in sorted(yx_dist):
	yx.append(yx_dist[identity])

print('Total identities: {}'.format(len(yx)))

yx1 = yx[:total_identities_1]
yx2 = yx[total_identities_1:total_identities_1+total_identities_2]
yx3 = yx[total_identities_1+total_identities_2:total_identities_1+total_identities_2+total_identities_3]

train1_xy = []
test1_xy = []
train2_xy = []
test2_xy = []
train3_xy = []
test3_xy = []

for i in range(total_identities_1):
	train_images = yx1[i][:280]
	test_images = yx1[i][280:]

	for image in train_images:
		train1_xy.append([image, i])

	for image in test_images:
		test1_xy.append([image, i])

for i in range(total_identities_2):
	train_images = yx2[i][:280]
	test_images = yx2[i][280:]

	for image in train_images:
		train2_xy.append([image, i])

	for image in test_images:
		test2_xy.append([image, i])

for i in range(total_identities_3):
	train_images = yx3[i][:280]
	test_images = yx3[i][280:]

	for image in train_images:
		train3_xy.append([image, i])

	for image in test_images:
		test3_xy.append([image, i])

np.random.shuffle(train1_xy)
np.random.shuffle(test1_xy)
np.random.shuffle(train2_xy)
np.random.shuffle(test2_xy)
np.random.shuffle(train3_xy)
np.random.shuffle(test3_xy)

for image, id in train1_xy:
	line = image + ' ' + str(id) + '\n'
	train1_anno_file.write(line)

for image, id in test1_xy:
	line = image + ' ' + str(id) + '\n'
	test1_anno_file.write(line)

print('Done, train 1: {}, test 1: {}'.format(len(train1_xy), len(test1_xy)))

for image, id in train2_xy:
	line = image + ' ' + str(id) + '\n'
	train2_anno_file.write(line)

for image, id in test2_xy:
	line = image + ' ' + str(id) + '\n'
	test2_anno_file.write(line)

print('Done, train 2: {}, test 2: {}'.format(len(train2_xy), len(test2_xy)))

for image, id in train3_xy:
	line = image + ' ' + str(id) + '\n'
	train3_anno_file.write(line)

for image, id in test3_xy:
	line = image + ' ' + str(id) + '\n'
	test3_anno_file.write(line)

print('Done, train 3: {}, test 3: {}'.format(len(train3_xy), len(test3_xy)))










