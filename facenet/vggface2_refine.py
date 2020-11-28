import cv2
import imagesize
from os import listdir


dataset_dir_path = '/Volumes/BAO/datasets/vggface2/train'
anno_file_path = '/Users/baulhoa/Documents/PythonProjects/datasets/faceid/refined.txt'

folders = listdir(dataset_dir_path)
total_identities = len(folders)
print('Total identities: {}'.format(total_identities))

selected_folders = []

for i in range(total_identities):
	folder = folders[i]
	if folder == '.DS_Store':
		continue

	images = listdir(dataset_dir_path + '/' + folder)
	image_count = len(images)

	if image_count >= 300:
		selected_folders.append(folder)

total_selected_identities = len(selected_folders)
anno_file = open(anno_file_path, 'w')
id = 0

for i in range(total_selected_identities):
	folder = selected_folders[i]

	images = listdir(dataset_dir_path + '/' + folder)
	image_count = len(images)

	selected_images = []

	for j in range(image_count):
		image = images[j]
		image_file_path = dataset_dir_path + '/' + folder + '/' + image
		w, h = imagesize.get(image_file_path)

		if h < 112 or w < 112:
			continue

		selected_images.append(image)

	total_selected_images = len(selected_images)

	if total_selected_images >= 300:
		for image in selected_images[:300]:
			line = folder + '/' + image + ' ' + str(id) + '\n'
			anno_file.write(line)

		id += 1

	print('-', end='')
	if i%100 == 99:
		print(id, round(i*100/total_selected_identities, 2), end='%\n')

print('Total selected identities: {}'.format(id))

anno_file.close()







	