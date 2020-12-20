DATASET_NUM_CLASSES = 19 + 1    # +1 for background class
DATASET_MEAN = [0.28689554, 0.32513303, 0.28389177]
DATASET_STD = [0.18696375, 0.19017339, 0.18720214]
# Maps labels to class indices
LABEL_MAPPING_DICT = \
{
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0, 10: 0,
    11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0, 19: 7,
    20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15,
    28: 16, 29: 0, 30: 0, 31: 17, 32: 18, 33: 19
}
assert min(LABEL_MAPPING_DICT.values()) == 0 and max(LABEL_MAPPING_DICT.values()) == (DATASET_NUM_CLASSES - 1), \
    "'LABEL_MAPPING_DICT' must contain mappings starting from 0 to {:d}!".format(DATASET_NUM_CLASSES - 1)
# Provides colors to each class to visualize segmentation maps
# NOTE: Color values from 'https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py'
    #       for all the classes where 'ignoreInEval' is False in 'labels' variable.
CLASS_RGB_COLOR = \
[
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), \
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), \
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), \
    (0, 80, 100), (0, 0, 230), (119, 11, 32)
]
assert len(CLASS_RGB_COLOR) == DATASET_NUM_CLASSES and all([len(x) == 3 for x in CLASS_RGB_COLOR]), \
    "'CLASS_RGB_COLOR' needs {:d} color values!".format(DATASET_NUM_CLASSES)