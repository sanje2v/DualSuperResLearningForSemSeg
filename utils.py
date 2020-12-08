
def check_version(version, major, minor):
    if type(version) == str:
        version = tuple(int(x) for x in version.split('.'))

    return version[0] >= major and version[1] >= minor


def getRGBColorFromClass(class_idx):
    # NOTE: Color values from 'https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py'
    #       for all the classes where 'ignoreInEval' is False in 'labels' variable.
    CLASS_RGB_COLOR = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), \
                       (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), \
                       (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), \
                       (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    assert len(CLASS_RGB_COLOR) == 19, "'CLASS_RGB_COLOR' needs 19 class color values!"
    return CLASS_RGB_COLOR[class_idx]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)