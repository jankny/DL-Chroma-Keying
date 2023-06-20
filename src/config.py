import os
from os import path
from os.path import join
from enum import Enum
from imutils import paths


class MODES(Enum):
    train = 1
    inference = 2


MODE = MODES.train

RANDOM_SEED = 42

# Image Paths from Maul
P_DATASET_BASE = "/home/jannes/uni/jk-masterarbeit/pascales-thesis/stick/DataSet"
P_DATASET_TRAIN = path.join(P_DATASET_BASE, "TrainingsSet", "ImagesScenes")
P_DATASET_PRETRAIN = path.join(P_DATASET_BASE, "TrainingsSet", "ImagesRand")
P_DATASET_EVAL = path.join(P_DATASET_BASE, "EvaluationSet")

DATA_SPLIT = "../data_split"
P_TRAIN_PATHS_GREEN = path.join(DATA_SPLIT, "P_TRAIN_PATHS_GREEN.json")
P_VAL_PATHS_GREEN = path.join(DATA_SPLIT, "P_VAL_PATHS_GREEN.json")
P_TEST_PATHS_GREEN = path.join(DATA_SPLIT, "P_TEST_PATHS_GREEN.json")

P_PRETRAIN_PATHS_GREEN = path.join(DATA_SPLIT, "P_PRETRAIN_PATHS_GREEN.json")
P_PRETRAIN_VAL_PATHS_GREEN = path.join(DATA_SPLIT, "P_PRETRAIN_VAL_PATHS_GREEN.json")
P_PRETRAIN_PATHS_BLUE = path.join(DATA_SPLIT, "P_PRETRAIN_PATHS_BLUE.json")
P_PRETRAIN_VAL_PATHS_BLUE = path.join(DATA_SPLIT, "P_PRETRAIN_VAL_PATHS_BLUE.json")

P_TRAIN_PATHS_BLUE = path.join(DATA_SPLIT, "P_TRAIN_PATHS_BLUE.json")
P_VAL_PATHS_BLUE = path.join(DATA_SPLIT, "P_VAL_PATHS_BLUE.json")
P_TEST_PATHS_BLUE = path.join(DATA_SPLIT, "P_TEST_PATHS_BLUE.json")

P_REGEX_REPLACE = ("SPILL_G", "GT", "SPILL_B")

# Evaluation without GT
P_EVAL_DIR = "/home/jannes/uni/jk-masterarbeit/pascales-thesis/System Evaluation"


# Image Paths from Adobe Deep Matting
DIM_DATASET_BASE = "/home/jannes/uni/jk-masterarbeit/data/Adobe_Deep_Matting_Dataset/Combined_Dataset"
DIM_DATASET_TRAIN = path.join(DIM_DATASET_BASE, "Training_set")
DIM_DATASET_TEST = path.join(DIM_DATASET_BASE, "Test_set")
DIM_TRAIN_PATHS = path.join(DIM_DATASET_TRAIN, "merged")
DIM_TEST_PATHS = path.join(DIM_DATASET_TEST, "merged")

DIM_REGEX_REPLACE = ("/merged/", "/gt/", "/gt/")

# Benchmark Paths
BENCHMARK_PATH = "/home/jannes/uni/jk-masterarbeit/green_benchmark/benchmark"
BENCHMARK_REGEX_REPLACE = ("/green/|/blue/|/red/", "/ground_truth/", "/blue/")
BLUE_PATHS = list(sorted(paths.list_images(join(BENCHMARK_PATH, "blue"))))
GREEN_PATHS = list(sorted(paths.list_images(join(BENCHMARK_PATH, "green"))))

VAL_SPLIT = 0.2

IMAGE_SIZE = (256, 256)
TRIMAP_KERNEL_SIZE = 30


NUM_EPOCHS = 80
BATCH_SIZE = 16

LOG_DIR = "../saves_logs"


class NNConfig():
    def __init__(self, trimap=False):
        self.NAME = ""
        self.NUM_EPOCHS = NUM_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

        self.INPUT_SKIP_LINK = False
        self.TRANSPOSED_CONVOLUTIONS = False
        self.TRIMAP_OUTPUT = trimap
        self.DIFFERENCE_OUTPUT = False
        self.COLOR_SPILL_MIXED_PIXELS_LOSS = False
        self.L1_LOSS = True
        self.INITIAL_CHANNEL_NUM = 11
        self.LEVELS = 3
        self.CHANNEL_FACTOR = 1.5

        self.DIM_PRETRAINIG = False
        self.DIM_NUM_EPOCHS = 30
        self.DIM_BG_AUGMENTATION = False

        self.MIXED_TRAIN_SET = False
        self.SEMANTIC_BRANCH = False
        self.AIM_RGB_ADDON = True
        self.AIM_RGB_BRANCH = False
        self.GRADIENT_LOSS = False

        self.BG_AUGMENTATION = False
        self.DATA_AUGMENTATION = False
        self.COLOR_AUGMENTATION = False
        self.CROP_AUGMENTATION = False

        self.IMAGE_SIZE = IMAGE_SIZE
        self.TRIMAP_KERNEL_SIZE = TRIMAP_KERNEL_SIZE

        self.RADAM = False
        self.RADAM_LR = 0.001
        self.RADAM_MIN_LR = 0.00001
        self.RADAM_WARMUP = 0.1

        self.A_WARMUP = False
        self.A_LR = 5e-4
        self.A_MIN_LR = 5e-6
        self.A_WARMUP_STEPS = 0.1

        self.DIM_TRAIN_WEIGHTS = None
        self.TRAIN_WEIGHTS = None
        self.LOSS_WEIGHTS = None

        self.COMMENT = ""
