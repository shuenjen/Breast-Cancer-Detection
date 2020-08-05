# This file is contributed by Adam Jaamour, Ashay Patel, and Shuen-Jen Chen

"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111

# basic
# VGG_IMG_SIZE = {
#     "HEIGHT": 224,
#     "WIDTH": 224
# }

# advance
VGG_IMG_SIZE = {
    "HEIGHT": 1024,
    "WIDTH": 1024
}

BATCH_SIZE = 8
EPOCH_1 = 150
EPOCH_2 = 50

# mini-MIAS
# CONV_CNT = 0
# DROPOUT = 0
# SAMPLING = "up"
# SAMPLING_TIMES = 1
# CLASS_WEIGHT = "x"
# CLASS_TYPE = "B-M"

# CBIS-DDSM
CONV_CNT = 2
DROPOUT = 0.5
SAMPLING = "x"
CLASS_WEIGHT = "x"
CLASS_TYPE = "B-M"

MODEL_SAVE_TIME=202008051516

# Variables set by command line arguments/flags
dataset = "mini-MIAS"   # The dataset to use.
cnn = "ResNet"             # CNN architecture
model = "basic"         # The model to use.
run_mode = "training"   # The type of running mode, either training or testing.
verbose_mode = False    # Boolean used to print additional logs for debugging purposes.

