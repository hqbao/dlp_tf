def settings(mode): 

	settings_for = mode 

	if settings_for == 'non-fpn-train':
		return {
			'start_example_index': 0,
			'num_of_train_examples': 2000,
			'num_of_validation_examples': 0,
			'asizes': [[91, 181], [128, 128], [181, 91]],
			'ishape': [1024, 1024, 3],
			'ssize': [32, 32],
			'resnet': [[16, 16, 64], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [1, 1]]],
			'max_num_of_rois': 7,
			'unified_roi_size': [7, 7],
			'rpn_head_dim': 256,
			'fc_denses': [1024, 1024],
			'iou_thresholds': [0.3, 0.5],
			'nsm_iou_threshold': 0.1,
			'nsm_score_threshold': 0.1,
			'num_of_samples': 64,
			'classes': ['face', 'none'],
			'mapping': {0: 0},
			'frame_mode': True,
			'num_of_epoches': 500,
			'base_block_trainable': True,
			'weight_loading': False,
			'dataset_anno_file_path': '../datasets/coco/annotations/instances_face.json',
			'dataset_image_dir_path': '../datasets/coco/images/face',
			'output_path': 'output'
		}

	if settings_for == 'non-fpn-inference':
		return {
			'start_example_index': 0,
			'num_of_train_examples': 0,
			'num_of_validation_examples': 0,
			'num_of_test_examples': 400,
			'asizes': [[91, 181], [128, 128], [181, 91]],
			'ishape': [1024, 1024, 3],
			'ssize': [32, 32],
			'resnet': [[16, 16, 64], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [1, 1]]],
			'max_num_of_rois': 7,
			'unified_roi_size': [7, 7],
			'rpn_head_dim': 256,
			'fc_denses': [1024, 1024],
			'iou_thresholds': [0.3, 0.5],
			'nsm_iou_threshold': 0.1,
			'nsm_score_threshold': 0.1,
			'num_of_samples': 64,
			'classes': ['face', 'none'],
			'mapping': {0: 0},
			'frame_mode': True,
			'dataset_anno_file_path': '../datasets/coco/annotations/instances_face.json',
			'dataset_image_dir_path': '../datasets/coco/images/face',
			'output_path': 'output'
		}

	if settings_for == 'fpn-train':
		return {
			'start_example_index': 0,
			'num_of_train_examples': 6700,
			'num_of_validation_examples': 0,
			'asizes': [
				[[32, 32]],
				[[64, 64]],
				[[128, 128]],
				[[256, 256]],
			],
			'ishape': [1024, 1024, 3],
			'ssizes': [[128, 128], [64, 64], [32, 32], [16, 16]],
			'resnet': [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]],
			'max_num_of_rois': 7,
			'unified_roi_size': [7, 7],
			'rpn_head_dim': 512,
			'fc_denses': [8],
			'iou_thresholds': [0.3, 0.5],
			'k0': 5,
			'top_down_pyramid_size': 512,
			'nsm_iou_threshold': 0.2,
			'nsm_score_threshold': 0.1,
			'num_of_samples': 64,
			'classes': ['face', 'none'],
			'mapping': {0: 0},
			'frame_mode': True,
			'num_of_epoches': 500,
			'base_block_trainable': True,
			'weight_loading': False,
			'dataset_anno_file_path': '../datasets/coco/annotations/instances_face.json',
			'dataset_image_dir_path': '../datasets/coco/images/face',
			'output_path': 'output'
		}

	if settings_for == 'fpn-inference':
		return {
			'start_example_index': 0,
			'num_of_train_examples': 0,
			'num_of_validation_examples': 0,
			'num_of_test_examples': 6700,
			'asizes': [
				[[32, 32]],
				[[64, 64]],
				[[128, 128]],
				[[256, 256]],
			],
			'ishape': [1024, 1024, 3],
			'ssizes': [[128, 128], [64, 64], [32, 32], [16, 16]],
			'resnet': [[64, 64, 256], [3, [2, 2]], [4, [2, 2]], [6, [2, 2]], [3, [2, 2]]],
			'max_num_of_rois': 7,
			'unified_roi_size': [7, 7],
			'rpn_head_dim': 512,
			'fc_denses': [8],
			'iou_thresholds': [0.3, 0.5],
			'k0': 5,
			'top_down_pyramid_size': 512,
			'nsm_iou_threshold': 0.2,
			'nsm_score_threshold': 0.1,
			'classes': ['face', 'none'],
			'mapping': {0: 0},
			'frame_mode': True,
			'dataset_anno_file_path': '../datasets/coco/annotations/instances_face.json',
			'dataset_image_dir_path': '../datasets/coco/images/face',
			'output_path': 'output'
		}


