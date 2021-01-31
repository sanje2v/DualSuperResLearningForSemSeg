import argparse
import numpy as np
import torch as t
import torchvision as tv
from tqdm.auto import tqdm

from utils import *
import settings
import consts


def calculate_dataset_mean_std(args):
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation from the dataset's specified split.")
    parser.add_argument('--dataset', required=True, nargs=2, metavar=('DATASET', 'SPLIT'), action=ValidateDatasetNameAndSplit, const=settings.DATASETS, help="Dataset and split to operate on")
    args = parser.parse_args(args)

    dataset_class = settings.DATASETS[args.dataset[0]]['class']
    dataset_path = settings.DATASETS[args.dataset[0]]['path']
    dataset_split = args.dataset[1]

    test_dataset = dataset_class(dataset_path,
                                 split=dataset_split,
                                 transforms=lambda img, seg: (tv.transforms.ToTensor()(img), np.empty(1)))
    test_loader = t.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=False,
                                          drop_last=False)

    # CAUTION: 'means = [[]] * consts.NUM_RGB_CHANNELS' is NOT the same as follows.
    means = [[] for _ in range(consts.NUM_RGB_CHANNELS)]
    stds = [[] for _ in range(consts.NUM_RGB_CHANNELS)]
    for input_img, _ in tqdm(test_loader,
                             desc='CALCULATING',
                             colour='green'):
        batch_mean_channels = t.mean(input_img, dim=(0, 2, 3))
        batch_std_channels = t.std(input_img, dim=(0, 2, 3))

        for channel in range(consts.NUM_RGB_CHANNELS):
            means[channel].append(batch_mean_channels[channel].item())
            stds[channel].append(batch_std_channels[channel].item())

    means = tuple(np.mean(means[channel]) for channel in range(consts.NUM_RGB_CHANNELS))
    stds = tuple(np.mean(stds[channel]) for channel in range(consts.NUM_RGB_CHANNELS))
    tqdm.write("\n---- RESULTS ---")
    tqdm.write("Avg. mean: ({0:.5f}, {1:.5f}, {2:.5f})".format(*means))
    tqdm.write("Avg. standard deviation: ({0:.5f}, {1:.5f}, {2:.5f})".format(*stds))