# This file is contributed by Adam Jaamour, Ashay Patel, and Shuen-Jen Chen

"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
VGG_IMG_SIZE = {
    "HEIGHT": 224,
    "WIDTH": 224
}
VGG_IMG_SIZE_LARGE = {
    "HEIGHT": 2048,
    "WIDTH": 2048
}
BATCH_SIZE = 8
EPOCH_1 = 150
EPOCH_2 = 50

# Variables set by command line arguments/flags
dataset = "mini-MIAS"   # The dataset to use.
cnn = "ResNet"             # CNN architecture
model = "basic"         # The model to use.
run_mode = "training"   # The type of running mode, either training or testing.
imagesize = "small"     # The size of input image, either small or large
verbose_mode = False    # Boolean used to print additional logs for debugging purposes.

