import argparse
import numpy as np
import torch as t
import torch.nn.functional as F
import torchvision as tv
from tqdm.auto import tqdm
from PIL import Image

import datasets.Cityscapes.settings as cityscapes_settings
from models import DSRL
from models.transforms import *
from utils import *
import settings
import consts


def show_augmented_inputs_targets(args):
    parser = argparse.ArgumentParser(description="Show augmented inputs fed to model and target outputs during training.")
    parser.add_argument('--dataset', required=True, nargs=2, metavar=('DATASET', 'SPLIT'), action=ValidateDatasetNameAndSplit, const=settings.DATASETS, help="Dataset and split to operate on")
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args(args)

    dataset_class = settings.DATASETS[args.dataset[0]]['class']
    dataset_path = settings.DATASETS[args.dataset[0]]['path']
    dataset_split = args.dataset[1]

    joint_transforms = JointCompose([JointRandomRotate(degrees=15.0, fill=(0, 0)),
                                     JointRandomCrop(min_scale=1.0, max_scale=3.5),
                                     JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                     JointColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                     JointHFlip(),
                                     # CAUTION: 'kernel_size' should be > 0 and odd integer
                                     JointRandomGaussianBlur(kernel_size=3, p=0.5),
                                     JointRandomGrayscale(p=0.1),
                                     JointNormalize(mean=cityscapes_settings.MEAN, std=cityscapes_settings.STD),
                                     JointScaledImage(new_size=settings.MODEL_INPUT_SIZE)])
    test_dataset = dataset_class(dataset_path,
                                 split=dataset_split,
                                 transforms=joint_transforms)
    test_loader = t.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=args.shuffle,
                                          num_workers=0,
                                          pin_memory=False,
                                          drop_last=False)

    print(INFO("Press ENTER to show next pair of input and output. Use CTRL+c to quit."))
    try:
        for i, ((_, input_image), target_map) in enumerate(test_loader):
            print("Creating visualization...")
            if input_image.shape[-2:] != target_map.shape[-2:]:
                input_image = F.interpolate(input_image, size=target_map.shape[-2:], mode='bilinear', align_corners=True)
            input_image = input_image.numpy()[0]
            input_image = np.array(cityscapes_settings.STD).reshape(consts.NUM_RGB_CHANNELS, 1, 1) * input_image +\
                          np.array(cityscapes_settings.MEAN).reshape(consts.NUM_RGB_CHANNELS, 1, 1)     # Unnormalize so that we can view the image
            input_image = np.clip(input_image * 255., a_min=0.0, a_max=255.).astype(np.uint8)
            target_map = target_map.numpy()[0]

            vis_image = make_input_output_visualization(input_image, target_map, cityscapes_settings.CLASS_RGB_COLOR)
            vis_image = vis_image.transpose((1, 2, 0))
            with Image.fromarray(vis_image, mode='RGB') as vis_image:
                vis_image.show(title=str(i))

            input()

    except KeyboardInterrupt:
        pass