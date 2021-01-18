import argparse
import numpy as np
import torch as t
import torchvision as tv
from tqdm.auto import tqdm as tqdm

from utils import *
import settings
import consts


def calculate_dataset_mean_std(args):
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation from the dataset's specified split.")
    parser.add_argument('--dataset-split', default='train', choices=['train', 'test', 'val'])
    args = parser.parse_args(args)

    dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                     split=args.dataset_split,
                                     mode='fine',
                                     target_type='semantic',
                                     transforms=lambda img, seg: (tv.transforms.ToTensor()(img), np.empty(1)))
    data_loader = t.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=False,
                                          drop_last=False)

    # CAUTION: 'means = [[]] * consts.NUM_RGB_CHANNELS' is NOT the same as follows.
    means = [[] for _ in range(consts.NUM_RGB_CHANNELS)]
    stds = [[] for _ in range(consts.NUM_RGB_CHANNELS)]
    for input_img, _ in tqdm(data_loader,
                             desc='CALCULATING',
                             colour='green'):
        batch_mean_channels = t.mean(input_img, dim=(0, 2, 3))
        batch_std_channels = t.std(input_img, dim=(0, 2, 3))

        for channel in range(consts.NUM_RGB_CHANNELS):
            means[channel].append(batch_mean_channels[channel])
            stds[channel].append(batch_std_channels[channel])

    means = tuple(np.mean(means[channel]) for channel in range(consts.NUM_RGB_CHANNELS))
    stds = tuple(np.mean(stds[channel]) for channel in range(consts.NUM_RGB_CHANNELS))
    tqdm.write("\n---- RESULTS ---")
    tqdm.write("Avg mean: ({0:.5f}, {1:.5f}, {2:.5f})".format(*means))
    tqdm.write("Avg standard deviation: ({0:.5f}, {1:.5f}, {2:.5f})".format(*stds))