#######################################################################################
##
## This script is used to just test out various functions of different Python modules
##
#######################################################################################


from utils import *

#from progress.bar import Bar as ProgressBar
#import time

#progressbar1 = ProgressBar('Training', max=100)

#for i in range(0, 100):
#    progressbar1.next()
#    time.sleep(.1)

#    if i % 20 == 0:
#        print()
#        progressbar2 = ProgressBar('Evaluation', max=50)
        
#        for j in range(0, 50):
#            progressbar2.next()
#            time.sleep(.1)
#        progressbar2.finish()
#        print("\033[A")
#progressbar1.finish()




#from tqdm.auto import tqdm
#import time

#bar_format = "{desc}: {percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}"

#with tqdm(total=100, desc="TRAINING", colour='green', position=0, leave=False, bar_format=bar_format) as p1:
#    for i in range(0, 100):
#        time.sleep(.1)

#        p1.set_postfix({"Postfix: ": str(i)})
#        p1.update()
        
#        #p1.set_postfix_str("Loss: %d" % i)

#        if i % 20 == 0:
#            for j in tqdm(range(0, 50), desc="VALIDATION", colour='yellow', position=1, leave=False, bar_format=bar_format):
#                time.sleep(.1)

#            tqdm.write('')
#            tqdm.write('')

#        p1.write("Epoch: %d" % i)

#from torch.utils import tensorboard as tb

#w=tb.SummaryWriter('logs/testing')

#w.add_text("INFO", "Information1", global_step=0)
#w.add_text("INFO", "Information2", global_step=0)

#w.close()




#import torch as t
#import torchvision as tv
#import numpy as np
#from PIL import Image
#from datasets.Cityscapes import settings as cityscapes_settings

#dataset = tv.datasets.Cityscapes('./datasets/Cityscapes/data', split='val', target_type='semantic', mode='fine')
##loader = t.utils.data.DataLoader(dataset, batch_size=4)

#min_,max_=(1000, -1000)
#for i in range(len(dataset)):
#    i,s = dataset[i]

#    min_=min(min_, np.amin(np.array(s)))
#    max_=max(max_, np.amax(np.array(s)))

#    #def t(x):
#    #    print(x)
#    #    return 0
#    #picnew = s.point(lambda x: cityscapes_settings.LABEL_MAPPING_DICT[x]) #Image.eval(s, t)

#print('min: {}, max: {}'.format(min_, max_))




#import argparse

#parser = argparse.ArgumentParser(description="Implementation of 'Dual Super Resolution Learning For Segmantic Segmentation' CVPR 2020 paper.")
#command_parser = parser.add_subparsers(title='commands', dest='command', required=True)

## Training commands
#train_parser = command_parser.add_parser('train', help='Train model for different stages')
#train_parser.add_argument('--resume_weight', default=None, type=str, help="Resume training with given weight file")
#train_parser.add_argument('--device', default='gpu', type=str.casefold, choices=['cpu', 'gpu'], help="Device to create model in")
#train_parser.add_argument('--num_workers', default=4, type=int, help="Number of workers for data loader")
#train_parser.add_argument('--val_interval', default=10, type=int, help="Epoch intervals after which to perform validation")
#train_parser.add_argument('--autosave_interval', default=5, type=int, help="Epoch intervals to auto save weights after in training")
#train_parser.add_argument('--autosave_history', default=5, type=int, help="Number of latest autosaved weights to keep while deleting old ones, 0 to disable autosave")
#train_parser.add_argument('--batch_size', default=4, type=int, help="Batch size to use for training and testing")
#train_parser.add_argument('--epochs', type=int, help="Number of epochs to train")
#train_parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
#train_parser.add_argument('--momentum', type=float, default=0.9, help="Momentum value for SGD")
#train_parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for SGD")
#train_parser.add_argument('--poly_power', type=float, default=0.9, help="Power for poly learning rate strategy")
#train_parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help="0: Train SSSR only\n1: Train SSSR+SISR\n2: Train SSSR+SISR with feature affinity")
#train_parser.add_argument('--w1', type=float, default=0.1, help="Weight for MSE loss")
#train_parser.add_argument('--w2', type=float, default=1.0, help="Weight for FA loss")
#train_parser.add_argument('--description', type=str, default='', help="Description of experiment to be saved in 'params.txt' with given commandline parameters")

## Testing commands
#test_parser = command_parser.add_parser('test', help='Test model')
#test_parser.add_argument('--out', default=None, type=str, help="Model weights")

#args = parser.parse_args()

#print(type(args))
#print(args)
#print(dir(args))
#print(args.__dict__)





#import termcolor

#def FATAL(text):
#    return termcolor.colored("FATAL: " + text, 'red', attrs=['reverse', 'blink'])

#print(FATAL("Something went wrong"))





#from PIL import Image

#img = Image.open('./datasets/Cityscapes/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')

#org_size = img.size    # CAUTION: For Image, size is (W, H) order in contrast to (H, W) for Tensor
#scale_factor = 3.5

#crop_width = int(1.0 / scale_factor * org_size[0])
#crop_height = int(1.0 / scale_factor * org_size[1])
#crop_x = (org_size[0] - crop_width) // 2
#crop_y = (org_size[1] - crop_height) // 2
#crop_box = [crop_x,\
#            crop_y,\
#            crop_x+crop_width,\
#            crop_y+crop_height]
#print(crop_box)

#img = img.resize(size=org_size, resample=Image.BILINEAR, box=crop_box)
#img.save('out.png')





import torch as t
import torchvision as tv
from models.transforms import *
from torchvision.transforms import ColorJitter
import numpy as np
import settings
from models import DSRL
import datasets.Cityscapes.settings as cityscapes_settings
from PIL import Image
from functools import partial

# SETTINGS
batch_size = 1
num_workers = 0

def funcTimeIt(label, *params):
    print(label + ": ", end='')
    timeit()
    return params
#train_joint_transforms = JointCompose([partial(funcTimeIt, 'start'),
#                                       JointRandomRotate(15.0, (0, 0)),
#                                       partial(funcTimeIt, 'JointRandomRotate'),
#                                       JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
#                                       partial(funcTimeIt, 'JointImageAndLabelTensor'),
#                                       JointRandomCrop(min_scale=1.0, max_scale=3.5),
#                                       partial(funcTimeIt, 'JointRandomCrop'),
#                                       lambda img, seg: (tv.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.4)(img), seg),
#                                       partial(funcTimeIt, 'ColorJitter'),
#                                       JointHFlip(),
#                                       partial(funcTimeIt, 'JointHFlip'),
#                                       # CAUTION: 'kernel_size' should be > 0 and odd integer
#                                       lambda img, seg: (tv.transforms.RandomApply([tv.transforms.GaussianBlur(kernel_size=3)], p=0.5)(img), seg),
#                                       partial(funcTimeIt, 'GaussianBlur'),
#                                       lambda img, seg: (tv.transforms.RandomGrayscale()(img), seg),
#                                       partial(funcTimeIt, 'RandomGrayscale'),
#                                       #lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
#                                       #partial(funcTimeIt, 'Normalize'),
#                                       lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg),
#                                       partial(funcTimeIt, 'DuplicateToScaledImageTransform')])
train_joint_transforms = JointCompose([partial(funcTimeIt, 'start'),
                                       JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                       partial(funcTimeIt, 'JointImageAndLabelTensor'),
                                       lambda img, seg: (ColorJitter2(brightness=0.4, contrast=0, saturation=0, hue=0.5)(img), seg),
                                       partial(funcTimeIt, 'ColorJitter'),
                                       lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg),
                                       partial(funcTimeIt, 'DuplicateToScaledImageTransform')])
val_joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                    #lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                    lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg)])
train_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                        split='train',
                                        mode='fine',
                                        target_type='semantic',
                                        transforms=train_joint_transforms)
train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                        split='val',
                                        mode='fine',
                                        target_type='semantic',
                                        transforms=val_joint_transforms)
val_loader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

data_loader = train_loader
for ((input_scaled, input_org), target) in data_loader:
    input_org = np.transpose(np.squeeze(input_org.cpu().numpy(), axis=0), (1, 2, 0))
    input_org = Image.fromarray(np.clip(input_org * 255., a_min=0.0, a_max=255.).astype(np.uint8)).convert('RGB')

    print('\n\n')

    input_org.show()

    #target = np.squeeze(target.cpu().numpy(), axis=0)
    #target_img = np.zeros((*target.shape, 3), dtype=np.uint8)

    #for y in range(target.shape[0]):
    #    for x in range(target.shape[1]):
    #        target_img[y, x, :] = cityscapes_settings.CLASS_RGB_COLOR[target[y, x]]

    #target_img = Image.fromarray(target_img).convert('RGB')

    #target_img.show()

    #blended = Image.blend(input_org, target_img, alpha=0.3)
    #blended.show()
    #input()






#import numpy as np

#def method_conf_mat(pred, target, num_classes): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
#    assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)!"
#    assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions!"

#    def _np_batch_bincount(arr, minlength):
#        return np.apply_along_axis(lambda x: np.bincount(x, minlength=minlength), axis=1, arr=arr)

#    pred = pred.reshape(pred.shape[0], -1)
#    target = target.reshape(target.shape[0], -1)

#    # Bincount of class detections
#    bincount_pred = _np_batch_bincount(pred, minlength=num_classes)
#    bincount_target = _np_batch_bincount(target, minlength=num_classes)

#    # Category matrix
#    category_matrix = target * num_classes + pred
#    bincount_category_matrix = _np_batch_bincount(category_matrix, minlength=(num_classes*num_classes))

#    # Confusion matrix
#    confusion_matrix = bincount_category_matrix.reshape((-1, num_classes, num_classes))

#    intersection = np.diagonal(confusion_matrix, axis1=1, axis2=2)
#    union = bincount_pred + bincount_target - intersection

#    with np.errstate(divide='warn', invalid='warn'): # NOTE: We ignore division by zero
#        return np.nanmean((intersection / union), axis=1)

#def method_hist(pred, target, num_classes):
#    assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)!"
#    assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions!"

#    def _np_batch_histogram(arr, bins, range_):
#        return np.apply_along_axis(lambda x: np.histogram(x, bins=bins, range=range_)[0], axis=1, arr=arr)

#    pred = pred.reshape(pred.shape[0], -1) + 1
#    target = target.reshape(target.shape[0], -1) + 1

#    intersection = pred * (pred == target)

#    area_intersection = _np_batch_histogram(intersection, bins=num_classes, range_=(1, num_classes))

#    area_pred = _np_batch_histogram(pred, bins=num_classes, range_=(1, num_classes))
#    area_target = _np_batch_histogram(target, bins=num_classes, range_=(1, num_classes))
#    area_union = area_pred + area_target - area_intersection

#    with np.errstate(divide='warn', invalid='warn'): # NOTE: We ignore division by zero
#        return np.nanmean((area_intersection / area_union), axis=1)

#batch_size = 2
#num_classes = 4

##pred = np.random.randint(1, num_classes, (batch_size, 3, 4))
##target = np.random.randint(1, num_classes, (batch_size, 3, 4))

#pred = np.zeros((batch_size, 3, 4), dtype=np.int32)
#target = np.zeros((batch_size, 3, 4), dtype=np.int32)

#pred[0, 1, 1] = 1

#print(method_conf_mat(pred, target, num_classes))
#print(method_hist(pred, target, num_classes))




#import numpy as np
#import torch as t

#def batch_intersection_union(predict, target, num_class, labeled):
#    predict = predict * labeled.long()
#    intersection = predict * (predict == target).long()

#    area_inter = t.histc(intersection.float(), bins=num_class, max=num_class, min=1)
#    area_pred = t.histc(predict.float(), bins=num_class, max=num_class, min=1)
#    area_lab = t.histc(target.float(), bins=num_class, max=num_class, min=1)
#    area_union = area_pred + area_lab - area_inter
#    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
#    return area_inter.cpu().numpy().sum() / area_union.cpu().numpy().sum()

#def batch_intersection_union2(pred, target, num_class, valid_labels_mask): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
#        assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)."
#        assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions."

#        pred = pred + 1
#        target = target + 1

#        pred = pred * valid_labels_mask
#        inter = pred * (pred == target)

#        area_pred, _ = np.histogram(pred, bins=num_class, range=(1, num_class))
#        area_inter, _ = np.histogram(inter, bins=num_class, range=(1, num_class))
#        area_target, _ = np.histogram(target, bins=num_class, range=(1, num_class))
#        area_union = area_pred + area_target - area_inter

#        assert (area_inter <= area_union).all(), "BUG CHECK: Intersection area should always be less than or equal to union area."

#        return (area_inter.sum() / area_union.sum())

#pred = np.array([[[0, 1, 3, 3, 4, 5], [2, 3, 1, 1, 3, 4]]], dtype=np.int64)
#target = np.array([[[0, 1, 2, 3, 4, 255], [2, 255, 1, 4, 255, 4]]], dtype=np.int64)
#labeled = np.array(target != 255, dtype=np.bool)

#print(batch_intersection_union(t.from_numpy(pred+1), t.from_numpy(target+1), 6, t.from_numpy(labeled)))
#print(batch_intersection_union2(pred, target, 6, labeled))