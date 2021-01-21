import os
import os.path
from tqdm.auto import tqdm
import numpy as np
import torch as t
import torchvision as tv

from models import DSRL
from models.transforms import *
from metrices import *
from utils import *
import settings
from datasets.Cityscapes import settings as cityscapes_settings


def benchmark(weights, dataset_split, device, device_obj, num_workers, batch_size, **other_args):
    # Run benchmark using specified weights and display results

    # Time keeper
    process_start_timestamp = datetime.now()

    # Create model and set to evaluation mode
    model = DSRL(stage=1).eval()

    # Load specified weights file
    model.load_state_dict(load_checkpoint_or_weights(weights, map_location=device_obj)['model_state_dict'], strict=True)

    # Copy the model into 'device_obj'
    model = model.to(device_obj)

    # Prepare data from CityScapes dataset
    os.makedirs(settings.CITYSCAPES_DATASET_DATA_DIR, exist_ok=True)
    if os.path.getsize(settings.CITYSCAPES_DATASET_DATA_DIR) == 0:
        raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'. Please refer to 'README.md'.".format(settings.CITYSCAPES_DATASET_DATA_DIR)))

    test_joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                          lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                          lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg)])
    test_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                          split=dataset_split,
                                          mode='fine',
                                          target_type='semantic',
                                          transforms=test_joint_transforms)
    test_loader = t.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=isCUDAdevice(device),
                                          drop_last=False)

    with t.no_grad(), tqdm(total=len(test_loader),
                           desc='BENCHMARKING',
                           colour='yellow',
                           position=0,
                           leave=False,
                           bar_format=settings.PROGRESSBAR_FORMAT) as progressbar:
        # Run benchmark
        CE_avg_loss = AverageMeter('CE Avg. Loss')
        miou = mIoU(num_classes=cityscapes_settings.DATASET_NUM_CLASSES)
        accuracy_mean = Accuracy()
        for ((input_scaled, _), target) in test_loader:
            SSSR_output, _, _, _ = model.forward(input_scaled.to(device_obj))
            SSSR_output = SSSR_output.detach().cpu()        # Bring back result to CPU memory

            # Calculate Cross entropy error
            CE_loss = t.nn.CrossEntropyLoss(ignore_index=cityscapes_settings.IGNORE_CLASS_LABEL)(SSSR_output, target)
            CE_avg_loss.update(CE_loss.item(), batch_size)

            # Prepare pred and target for metrices to process
            SSSR_output = SSSR_output.numpy()
            pred = np.argmax(SSSR_output, axis=1)               # Convert probabilities across dimensions to class label in one 2-D grid
            target = target.detach().cpu().numpy()

            # Remove invalid background class from evaluation
            valid_labels_mask = (target != cityscapes_settings.IGNORE_CLASS_LABEL)    # Boolean mask

            # Calculate metrices for this batch
            miou.update(pred, target, valid_labels_mask)
            accuracy_mean.update(pred, target, valid_labels_mask)

            progressbar.update()

    total_miou = miou() * 100.0
    total_mean_accuracy = accuracy_mean() * 100.0
    tqdm.write("-------- RESULTS --------")
    tqdm.write("Avg. Cross Entropy Error: {:.3f}".format(CE_avg_loss.avg))
    tqdm.write("mIoU %: {:.2f}".format(total_miou))
    tqdm.write("Mean Accuracy %: {:.2f}".format(total_mean_accuracy))

    # Save benchmark result to output directories in 'benchmark.txt'
    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    output_benchmark_filename = os.path.join(settings.OUTPUTS_DIR, 'benchmark.txt')
    with open(output_benchmark_filename, 'w') as benchmark_file:
        benchmark_file.write("Benchmarking results on Cityscapes dataset's {:s} split\n\n".format(dataset_split))
        benchmark_file.write("On: {:s}\n".format(process_start_timestamp.strftime("%c")))
        benchmark_file.write("Weights file: {:s}\n\n".format(weights))
        benchmark_file.write("Avg. Cross Entropy Error: {:.3f}".format(CE_avg_loss.avg))
        benchmark_file.write("mIoU %: {:.2f}".format(total_miou))
        benchmark_file.write("Mean Accuracy %: {:.2f}".format(total_mean_accuracy))