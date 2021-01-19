import argparse
import numpy as np
import torch as t
import torchvision as tv
from tqdm.auto import tqdm as tqdm
from PIL import Image

import datasets.Cityscapes.settings as cityscapes_settings
from models import DSRL
from models.transforms import *
from utils import *
import settings
import consts


def show_augmented_inputs_targets(args):
    parser = argparse.ArgumentParser(description="Show augmented inputs fed to model and target outputs during training.")
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args(args)

    joint_transforms = JointCompose([JointRandomRotate(degrees=15.0, fill=(0, 0)),
                                     JointRandomCrop(min_scale=1.0, max_scale=3.5),
                                     JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                     lambda img, seg: (ColorJitter2(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)(img), seg),
                                     JointHFlip(),
                                     # CAUTION: 'kernel_size' should be > 0 and odd integer
                                     lambda img, seg: (tv.transforms.RandomApply([tv.transforms.GaussianBlur(kernel_size=3)], p=0.5)(img), seg),
                                     lambda img, seg: (tv.transforms.RandomGrayscale(p=0.1)(img), seg),
                                     lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                     lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg)])
    dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                     split='train',
                                     mode='fine',
                                     target_type='semantic',
                                     transforms=joint_transforms)
    data_loader = t.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=args.shuffle,
                                          num_workers=0,
                                          pin_memory=False,
                                          drop_last=False)

    print(INFO("Press ENTER to show next pair of input and output. Use CTRL+c to quit."))
    try:
        for i, ((_, input_image), target_map) in enumerate(data_loader):
            input_image = input_image.numpy()[0]
            input_image = (input_image.transpose((1, 2, 0)) * cityscapes_settings.DATASET_STD) + cityscapes_settings.DATASET_MEAN
            input_image = input_image.transpose((2, 0, 1))
            input_image = np.clip(input_image * 255., a_min=0.0, a_max=255.).astype(np.uint8)
            target_map = target_map.numpy()[0]

            vis_image = make_input_output_visualization(input_image, target_map, cityscapes_settings.CLASS_RGB_COLOR)
            vis_image = vis_image.transpose((1, 2, 0))
            with Image.fromarray(vis_image, mode='RGB') as vis_image:
                vis_image.show(title=str(i))

            input()

    except KeyboardInterrupt:
        pass