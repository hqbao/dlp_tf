

anno_file_path = 'anno/celeba_id_landmark_anno.txt'
anno1_file_path = '../datasets/CelebA/Anno/identity_CelebA.txt'
anno2_file_path = '../datasets/CelebA/Anno/list_landmarks_align_celeba.txt'

anno_file = open(anno_file_path, 'w')
anno1_file = open(anno1_file_path, 'r')
anno2_file = open(anno2_file_path, 'r')

lines1 = anno1_file.readlines()
lines2 = anno2_file.readlines()

del lines2[0]
del lines2[0]

total_images = len(lines1)

if total_images != len(lines2):
	exit(0)

for i in range(total_images):
	line1 = lines1[i][:-1]
	anno1 = line1.split(' ')

	line2 = lines2[i][:-1]
	line2 = line2.replace('    ', ' ')
	line2 = line2.replace('   ', ' ')
	line2 = line2.replace('  ', ' ')
	anno2 = line2.split(' ')

	anno1.extend(anno2[1:])

	line = ' '.join(anno1) + '\n'
	anno_file.write(line)

anno_file.close()
anno1_file.close()
anno2_file.close()





