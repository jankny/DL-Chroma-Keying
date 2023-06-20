import re
import json
import random
import config
import numpy as np
from imutils import paths

image_paths = list(paths.list_images(config.P_DATASET_TRAIN))
images_green = [p for p in image_paths if "SPILL_G" in p and "bg_color" not in p]
images_blue  = [p for p in image_paths if "SPILL_B" in p and "bg_color" not in p]

image_paths_pretrain = list(paths.list_images(config.P_DATASET_PRETRAIN))
images_green_pretrain = [p for p in image_paths_pretrain if "SPILL_G" in p and "bg_color" not in p]
images_blue_pretrain  = [p for p in image_paths_pretrain if "SPILL_B" in p and "bg_color" not in p]

image_paths_test = list(paths.list_images(config.P_DATASET_EVAL))
images_green_test = [p for p in image_paths_test if "SPILL_G" in p and "bg_color" not in p]
images_blue_test  = [p for p in image_paths_test if "SPILL_B" in p and "bg_color" not in p]

scene_name_re = re.compile("[^/]+(?=_\d)")
scene_names = set([scene_name_re.search(p).group() for p in images_blue])
scene_names_pretrain = set([scene_name_re.search(p).group() for p in images_blue_pretrain])


images_green_val = []
images_green_train = []
images_blue_val = []
images_blue_train = []
for scene_name in scene_names:
	imgs_green = sorted([p for p in images_green if scene_name in p])
	imgs_blue = sorted([p for p in images_blue if scene_name in p])

	i = int(len(imgs_green) * config.VAL_SPLIT)
	images_green_val.extend(imgs_green[:i])
	images_green_train.extend(imgs_green[i:])

	i = int(len(imgs_blue) * config.VAL_SPLIT)
	images_blue_val.extend(imgs_blue[:i])
	images_blue_train.extend(imgs_blue[i:])

images_green_val_pretrain = []
images_green_train_pretrain = []
images_blue_val_pretrain = []
images_blue_train_pretrain = []
for scene_name in scene_names_pretrain:
	imgs_green = sorted([p for p in images_green_pretrain if scene_name in p])
	imgs_blue = sorted([p for p in images_blue_pretrain if scene_name in p])

	i = int(len(imgs_green) * config.VAL_SPLIT)
	images_green_val_pretrain.extend(imgs_green[:i])
	images_green_train_pretrain.extend(imgs_green[i:])

	i = int(len(imgs_blue) * config.VAL_SPLIT)
	images_blue_val_pretrain.extend(imgs_blue[:i])
	images_blue_train_pretrain.extend(imgs_blue[i:])

datasets = [
	("training_green", images_green_train, config.P_TRAIN_PATHS_GREEN),
	("validation_green", images_green_val, config.P_VAL_PATHS_GREEN),
	("testing_green", images_green_test, config.P_TEST_PATHS_GREEN),

	("training_blue", images_blue_train, config.P_TRAIN_PATHS_BLUE),
	("validation_blue", images_blue_val, config.P_VAL_PATHS_BLUE),
	("testing_blue", images_blue_test, config.P_TEST_PATHS_BLUE),

	("pretraining_green", images_green_train_pretrain, config.P_PRETRAIN_PATHS_GREEN),
	("pretraining_val_green", images_green_val_pretrain, config.P_PRETRAIN_VAL_PATHS_GREEN),
	("pretraining_blue", images_blue_train_pretrain, config.P_PRETRAIN_PATHS_BLUE),
	("pretraining_val_blue", images_blue_val_pretrain, config.P_PRETRAIN_VAL_PATHS_BLUE),
]

for name, paths, file in datasets:
	with open(file, 'w') as f:
		json.dump(paths, f)
