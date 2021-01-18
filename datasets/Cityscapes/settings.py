from .consts import NUM_RGB_CHANNELS

DATASET_NUM_CLASSES = 19
# NOTE: Computed using 'python run_script.py calculate_dataset_mean_std --dataset-split train'
DATASET_MEAN = (0.28690, 0.32513, 0.28389)
DATASET_STD = (0.17614, 0.18099, 0.17772)
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
{
    0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70), 3: (102, 102, 156), 4: (190, 153, 153),\
    5: (153, 153, 153), 6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35), 9: (152, 251, 152),\
    10: (70, 130, 180), 11: (220, 20, 60), 12: (255, 0, 0), 13: (0, 0, 142), 14: (0, 0, 70), 15: (0, 60, 100),\
    16: (0, 80, 100), 17: (0, 0, 230), 18: (119, 11, 32), IGNORE_CLASS_LABEL: (0, 0, 0)
}
assert len(CLASS_RGB_COLOR.items()) == DATASET_NUM_CLASSES + 1 and all([len(x) == NUM_RGB_CHANNELS for x in CLASS_RGB_COLOR.values()]),\
    "'CLASS_RGB_COLOR' needs {:d} color values with three RGB color values!".format(DATASET_NUM_CLASSES)