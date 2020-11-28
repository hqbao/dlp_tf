import cv2
import numpy as np
import os


dataset_dir_path = '/Users/baulhoa/Documents/PythonProjects/datasets/vggface2/train'
new_dataset_dir_path = '/Users/baulhoa/Documents/PythonProjects/datasets/vggface2/train_refined_resized'
anno_file_path = 'anno/vggface2_refined_anno.txt'
ishape = [142, 128, 3]

anno_file = open(anno_file_path, 'r')
lines = anno_file.readlines()
total_images = len(lines)

print('Total images: {}'.format(total_images))

for i in range(total_images):
	line = lines[i][:-1]
	anno = line.split(' ')
	id_folder, file_name = anno[0].split('/')
	image_file_path = dataset_dir_path + '/' + id_folder + '/' + file_name
	x = cv2.imread(image_file_path)
	x = cv2.resize(x, dsize=(ishape[1], ishape[0]), interpolation=cv2.INTER_CUBIC)
	x = np.clip(x, 0, 255)

	id_folder_path = new_dataset_dir_path + '/' + id_folder
	if not os.path.exists(id_folder_path):
		os.mkdir(id_folder_path)

	cv2.imwrite(id_folder_path + '/' + file_name, x)

	if i%100 == 99:
		print('-', end='')
	if i%10000 == 9999:
		print(round(i*100/total_images, 2), end='%\n')