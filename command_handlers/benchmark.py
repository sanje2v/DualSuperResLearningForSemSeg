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


@t.no_grad()
def benchmark(weights, dataset, device, num_workers, batch_size, **other_args):
    # Run benchmark using specified weights and display results

    # Time keeper
    process_start_timestamp = datetime.now()

    device_obj = t.device('cuda' if isCUDAdevice(device) else device)

    # Create model, set to evaluation mode and copy to device
    model = DSRL(stage=1, dataset_settings=dataset['settings']).eval().to(device_obj)

    # Load specified weights file
    model.load_state_dict(load_checkpoint_or_weights(weights, map_location=device_obj)['model_state_dict'], strict=False)

    # Prepare data from CityScapes dataset
    os.makedirs(dataset['path'], exist_ok=True)
    if os.path.getsize(dataset['path']) == 0:
        raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'. Please refer to 'README.md'.".format(dataset['path'])))

    test_joint_transforms = JointCompose([JointImageAndLabelTensor(dataset['settings'].LABEL_MAPPING_DICT),
                                          JointNormalize(mean=dataset['settings'].MEAN, std=dataset['settings'].STD),
                                          JointScaledImage(new_img_sizes=(settings.MODEL_INPUT_SIZE, settings.MODEL_OUTPUT_SIZE), new_seg_size=settings.MODEL_OUTPUT_SIZE)])
    test_dataset = dataset['class'](dataset['path'],
                                    split=dataset['split'],
                                    transforms=test_joint_transforms)
    test_loader = t.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=isCUDAdevice(device),
                                          drop_last=False)

    with tqdm(total=len(test_loader),
                desc='BENCHMARKING',
                colour='yellow',
                position=0,
                leave=False,
                bar_format=settings.PROGRESSBAR_FORMAT) as progressbar:
        # Run benchmark
        CE_avg_loss = AverageMeter()
        miou = mIoU(num_classes=dataset['settings'].NUM_CLASSES)
        accuracy_mean = Accuracy()

        for ((input_image, _), (target, _)) in test_loader:
            SSSR_output, _, _, _ = model.forward(input_image.to(device_obj))
            SSSR_output = SSSR_output.detach().cpu()        # Bring back result to CPU memory

            # Calculate Cross entropy error
            CE_loss = t.nn.CrossEntropyLoss(ignore_index=dataset['settings'].IGNORE_CLASS_LABEL)(SSSR_output, target.long())
            CE_avg_loss.update(CE_loss.item(), batch_size)

            # Prepare pred and target for metrices to process
            SSSR_output = SSSR_output.numpy()
            pred = np.argmax(SSSR_output, axis=1)           # Convert probabilities across dimensions to class label in one 2-D grid
            target = target.detach().cpu().numpy()

            # Remove invalid background class from evaluation
            valid_labels_mask = (target != dataset['settings'].IGNORE_CLASS_LABEL)    # Boolean mask

            # Calculate metrices for this batch
            miou.update(pred, target, valid_labels_mask)
            accuracy_mean.update(pred, target, valid_labels_mask)

            progressbar.update()

    print("-------- RESULTS --------")
    print("Avg. Cross Entropy Error: {:.3f}".format(CE_avg_loss()))
    print("mIoU %: {:.2f}".format(miou()))
    print("Mean Accuracy %: {:.2f}".format(accuracy_mean()))

    # Save benchmark result to output directories in 'benchmark.txt'
    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    output_benchmark_filename = os.path.join(settings.OUTPUTS_DIR, 'benchmark.txt')
    with open(output_benchmark_filename, 'w') as benchmark_file:
        benchmark_file.write("Benchmarking results on Cityscapes dataset's {:s} split\n\n".format(dataset['split']))
        benchmark_file.write("On: {:s}\n".format(process_start_timestamp.strftime("%c")))
        benchmark_file.write("Weights file: {:s}\n\n".format(weights))
        benchmark_file.write("Avg. Cross Entropy Error: {:.3f}".format(CE_avg_loss()))
        benchmark_file.write("mIoU %: {:.2f}".format(miou()))
        benchmark_file.write("Mean Accuracy %: {:.2f}".format(accuracy_mean()))