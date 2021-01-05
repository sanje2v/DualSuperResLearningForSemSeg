from .consts import NUM_RGB_CHANNELS

# Ref: https://github.com/inferno-pytorch/inferno/blob/master/inferno/io/box/cityscapes.py
DATASET_NUM_CLASSES = 19
DATASET_MEAN = [0.485, 0.456, 0.406]
DATASET_STD = [0.229, 0.224, 0.225]
# Maps labels to class indices
IGNORE_CLASS_LABEL = 255
LABEL_MAPPING_DICT =\
{
    0: IGNORE_CLASS_LABEL, 1: IGNORE_CLASS_LABEL, 2: IGNORE_CLASS_LABEL, 3: IGNORE_CLASS_LABEL,
    4: IGNORE_CLASS_LABEL, 5: IGNORE_CLASS_LABEL, 6: IGNORE_CLASS_LABEL, 7: 0, 8: 1,
    9: IGNORE_CLASS_LABEL, 10: IGNORE_CLASS_LABEL, 11: 2, 12: 3, 13: 4, 14: IGNORE_CLASS_LABEL,
    15: IGNORE_CLASS_LABEL, 16: IGNORE_CLASS_LABEL, 17: 5, 18: IGNORE_CLASS_LABEL, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: IGNORE_CLASS_LABEL,
    30: IGNORE_CLASS_LABEL, 31: 16, 32: 17, 33: 18, -1: IGNORE_CLASS_LABEL
}

# Provides colors to each class to visualize segmentation maps
# NOTE: Color values from 'https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py'
#       for all the classes where 'ignoreInEval' is False in 'labels' variable.
CLASS_RGB_COLOR =\
[
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),\
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),\
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),\
    (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)
]
assert len(CLASS_RGB_COLOR) == (DATASET_NUM_CLASSES + 1) and all([len(x) == NUM_RGB_CHANNELS for x in CLASS_RGB_COLOR]),\
    "'CLASS_RGB_COLOR' needs {:d} color values with three RGB color values!".format(DATASET_NUM_CLASSES)