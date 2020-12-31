import sys
import os
import os.path
import shutil
import argparse
import termcolor
from tqdm.auto import tqdm as tqdm
import numpy as np
import torch as t
import torchvision as tv
from torch.utils import tensorboard as tb
from datetime import datetime, timedelta
from PIL import Image, ImageOps

from models import DSRLSS
from models.schedulers import PolynomialLR
from models.losses import FALoss
from models.transforms import *
from metrices import AverageMeter, mIoU
from utils import *
import consts
import settings
from datasets.Cityscapes import settings as cityscapes_settings



def do_train_val(do_train: bool, model, device, batch_size, stage, data_loader, w1=None, w2=None, optimizer=None, scheduler=None):
    # Set model to either training or testing mode
    model.train(mode=do_train)

    # Losses to report
    CE_avg_loss = AverageMeter('CE Avg. Loss')
    MSE_avg_loss = AverageMeter('MSE Avg. Loss')
    FA_avg_loss = AverageMeter('FA Avg. Loss')
    Avg_loss = AverageMeter('Avg. Loss')

    with t.set_grad_enabled(mode=do_train), tqdm(total=len(data_loader),
                                                 desc='TRAINING' if do_train else 'VALIDATION',
                                                 colour='green' if do_train else 'yellow',
                                                 position=0 if do_train else 1,
                                                 leave=False,
                                                 bar_format=settings.PROGRESSBAR_FORMAT) as progressbar:
        for ((input_scaled, input_org), target) in data_loader:
            # SANITY CHECK: Check data doesn't have any 'NaN' values
            assert not (t.isnan(input_scaled).any().item()),\
                FATAL("'input_scaled' contains 'NaN' values")
            assert not (False if input_org is None else t.isnan(input_org).any().item()),\
                FATAL("'input_org' contains 'NaN' values")
            assert not (t.isnan(target).any().item()),\
                FATAL("'target' contains 'NaN' values")

            input_scaled = input_scaled.to(device)
            input_org = (None if stage == 1 else input_org.to(device))
            target = target.to(device)
            if do_train:
                optimizer.zero_grad()

            SSSR_output, SISR_output, SSSR_transform_output, SISR_transform_output = model.forward(input_scaled)
            # SANITY CHECK: Check network outputs doesn't have any 'NaN' values
            assert not (t.isnan(SSSR_output).any().item()),\
                FATAL("SSSR network output contains 'NaN' values and so cannot continue. Exiting.")
            assert not (False if SISR_output is None else t.isnan(SISR_output).any().item()),\
                FATAL("SISSR network output contains 'NaN' values and so cannot continue. Exiting.")

            CE_loss = t.nn.CrossEntropyLoss()(SSSR_output, target)
            MSE_loss = (w1 * t.nn.MSELoss()(SISR_output, input_org)) if stage > 1 else t.tensor(0., requires_grad=False)
            FA_loss = (w2 * FALoss()(SSSR_transform_output, SISR_transform_output)) if stage > 2 else t.tensor(0., requires_grad=False)
            loss = CE_loss + MSE_loss + FA_loss

            if do_train:
                loss.backward()     # Backpropagate
                optimizer.step()    # Increment global step

            # Convert loss tensors to float on CPU memory
            CE_loss = CE_loss.item()
            MSE_loss = MSE_loss.item() if stage > 1 else None
            FA_loss = FA_loss.item() if stage > 2 else None
            loss = loss.item()

            # Compute averages for losses
            CE_avg_loss.update(CE_loss, batch_size)
            if stage > 1:
                MSE_avg_loss.update(MSE_loss, batch_size)
                if stage > 2:
                    FA_avg_loss.update(FA_loss, batch_size)
                Avg_loss.update(loss, batch_size)

            # Add loss information to progress bar
            log_string = []
            log_string.append("CE: {:.4f}".format(CE_loss))
            if stage > 1:
                log_string.append("MSE: {:.4f}".format(MSE_loss))
                if stage > 2:
                    log_string.append("FA: {:.4f}".format(FA_loss))
                log_string.append("Total: {:.3f}".format(loss))
            log_string = ', '.join(log_string)
            progressbar.set_postfix_str("Losses [{0}]".format(log_string))
            progressbar.update()

        # Show learning rate and average losses before ending epoch
        log_string = []
        log_string.append("Learning Rate: {:6f}".format(scheduler.get_last_lr()[0]) if do_train else "Validation results:")
        log_string.append("Avg. CE: {:.4f}".format(CE_avg_loss.avg))
        if stage > 1:
            log_string.append("Avg. MSE: {:.4f}".format(MSE_avg_loss.avg))
            if stage > 2:
                log_string.append("Avg. FA: {:.4f}".format(FA_avg_loss.avg))
            log_string.append("Total Avg. Loss: {:.3f}".format(Avg_loss.avg))
        log_string = ', '.join(log_string)
        tqdm.write(log_string)

    return CE_avg_loss, MSE_avg_loss, FA_avg_loss, Avg_loss


def write_params_file(filename, *list_params):
    with open(filename, mode='w') as params_file:
        for params_str in list_params:
            if params_str:
                params_file.write(params_str)
                params_file.write('\n')     # NOTE: '\n' here automatically converts it to newline for the current platform


def load_checkpoint_or_weights(filename):
    return t.load(filename)


def save_checkpoint(dir, filename, **checkpoint_vars):
    os.makedirs(dir, exist_ok=True)
    t.save(checkpoint_vars, os.path.join(dir, filename))


def save_weights(dir, filename, model):
    os.makedirs(dir, exist_ok=True)
    t.save({'model_state_dict': model.state_dict()}, os.path.join(dir, filename))


def main(command,
         device=None,
         num_workers=None,
         val_interval=None,
         checkpoint=None,
         checkpoint_interval=None,
         checkpoint_history=None,
         init_weights=None,
         batch_size=None,
         epochs=None,
         learning_rate=None,
         momentum=None,
         weights_decay=None,
         poly_power=None,
         stage=None,
         w1=None,
         w2=None,
         description=None,
         image_file=None,
         weights=None,
         src_weights=None,
         dest_weights=None,
         dataset_split=None):

    # Time keeper
    process_start_timestamp = datetime.now()

    if command == 'resume_train':
        checkpoint_dict = load_checkpoint_or_weights(checkpoint)

        device = checkpoint_dict['device']
        num_workers = checkpoint_dict['num_workers']
        val_interval = checkpoint_dict['val_interval']
        checkpoint_interval = checkpoint_dict['checkpoint_interval']
        checkpoint_history = checkpoint_dict['checkpoint_history']
        init_weights = checkpoint_dict['init_weights']
        batch_size = checkpoint_dict['batch_size']
        epochs = checkpoint_dict['epochs']
        learning_rate = checkpoint_dict['learning_rate']
        momentum = checkpoint_dict['momentum']
        weights_decay = checkpoint_dict['weights_decay']
        poly_power = checkpoint_dict['poly_power']
        stage = checkpoint_dict['stage']
        w1 = checkpoint_dict['w1']
        w2 = checkpoint_dict['w2']
        description = checkpoint_dict['description']

    if device:
        # Device to perform calculation in
        if device.startswith(('gpu', 'cuda')) and not t.cuda.is_available():
            raise Exception("CUDA is not available to use GPU!")
        target_device = t.device('cuda' if device == 'gpu' else device)

    if command in ['train', 'resume_train']:
        # Training and Validation on dataset mode

        # Prevent system from entering sleep state so that long training session is not interrupted
        if prevent_system_sleep():
            tqdm.write(INFO("System will NOT be allowed to sleep until this training is complete/interrupted."))
        else:
            tqdm.write(CAUTION("Please make sure system is NOT configured to sleep on idle! Sleep mode will pause training."))

        # Create model according to stage
        model = DSRLSS(stage)
        if command == 'resume_train':
            model.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
        else:
            # Load initial weight, if any
            if init_weights:
                model.load_state_dict(load_weights_or_checkpoint(init_weights)['model_state_dict'], strict=False)

            # Load checkpoint from previous stage, if not the first stage
            if stage > 1:
                weights_dict = load_checkpoint_or_weights(os.path.join(settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE))
                model.load_state_dict(weights_dict['model_state_dict'], strict=False)

        # Copy the model into 'target_device' memory
        model = model.to(target_device)

        # Prepare data from CityScapes dataset
        os.makedirs(settings.CITYSCAPES_DATASET_DATA_DIR, exist_ok=True)
        if os.path.getsize(settings.CITYSCAPES_DATASET_DATA_DIR) == 0:
            raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'.".format(settings.CITYSCAPES_DATASET_DATA_DIR)))

        train_joint_transforms = JointCompose([JointRandomCrop(min_scale=1.0, max_scale=3.5),
                                               JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                               lambda img, seg: (tv.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)(img), seg),
                                               JointHFlip(),
                                               # CAUTION: 'kernel_size' should be > 0 and odd integer
                                               lambda img, seg: (tv.transforms.RandomApply([tv.transforms.GaussianBlur(kernel_size=3)], p=0.5)(img), seg),
                                               lambda img, seg: (tv.transforms.RandomGrayscale()(img), seg),
                                               lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                               lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRLSS.MODEL_INPUT_SIZE)(img), seg)])
        val_joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                             lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                             lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRLSS.MODEL_INPUT_SIZE)(img), seg)])

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

        # Make sure proper log directories exist
        train_logs_dir = settings.LOGS_DIR.format(stage=stage, mode='train')
        val_logs_dir = settings.LOGS_DIR.format(stage=stage, mode='val')
        os.makedirs(train_logs_dir, exist_ok=True)
        os.makedirs(val_logs_dir, exist_ok=True)

        # Write training parameters provided to params.txt log file
        write_params_file(os.path.join(train_logs_dir, settings.PARAMS_FILE),
                          "Timestamp: {:s}".format(process_start_timestamp.strftime("%c")),
                          "Device: {:s}".format(device),
                          "No. of workers: {:d}".format(num_workers),
                          "Validation interval: {:d}".format(val_interval),
                          "Checkpoint interval: {:d}".format(checkpoint_interval),
                          "Checkpoint history: {:d}".format(checkpoint_history),
                          "Initial weights: {:s}".format(init_weights) if init_weights else None,
                          "Resume checkpoint: {:s}".format(checkpoint) if checkpoint else None,
                          "Batch size: {:d}".format(batch_size),
                          "Epochs: {:d}".format(epochs),
                          "Learning rate: {:f}".format(learning_rate),
                          "Momentum: {:f}".format(momentum),
                          "Weights decay: {:f}".format(weights_decay),
                          "Poly power: {:f}".format(poly_power),
                          "Stage: {:d}".format(stage),
                          "Loss Weight 1: {:.4f}".format(w1) if stage > 1 else None,
                          "Loss Weight 2: {:.4f}".format(w2) if stage > 2 else None,
                          "Description: {:s}".format(description) if description else None)

        # Start training and validation
        with tb.SummaryWriter(log_dir=train_logs_dir) as train_logger,\
             tb.SummaryWriter(log_dir=val_logs_dir) as val_logger:

            # Training optimizer and schedular
            optimizer = t.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weights_decay)
            if command == 'resume_train':
                optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                starting_epoch = checkpoint_dict['epoch'] if command == 'resume_train' else 0

            scheduler = PolynomialLR(optimizer,
                                     max_decay_steps=epochs,
                                     end_learning_rate=0.001,
                                     power=poly_power,
                                     last_epoch=(starting_epoch-1))

            # Start training and then validation after specific intervals
            train_logger.add_text("INFO", "Training started on {:s}".format(process_start_timestamp.strftime("%c")), 1)
            log_string = "\n################################# Stage {:d} training STARTED #################################".format(stage)
            tqdm.write(log_string)

            for epoch in range((starting_epoch + 1), (epochs + 1)):
                log_string = "\nEPOCH {0:d}/{1:d}".format(epoch, epochs)
                tqdm.write(log_string)

                # Do training for this epoch
                CE_train_avg_loss, \
                MSE_train_avg_loss, \
                FA_train_avg_loss, \
                Avg_train_loss = do_train_val(do_train=True,
                                              model=model,
                                              device=target_device,
                                              batch_size=batch_size,
                                              stage=stage,
                                              data_loader=train_loader,
                                              w1=w1,
                                              w2=w2,
                                              optimizer=optimizer,
                                              scheduler=scheduler)

                # Log training losses for this epoch to TensorBoard
                train_logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_train_avg_loss.avg, epoch)
                if stage > 1:
                    train_logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_train_avg_loss.avg, epoch)
                    if stage > 2:
                        train_logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_train_avg_loss.avg, epoch)
                    train_logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_train_loss.avg, epoch)

                # Log learning rate for this epoch to TensorBoard
                train_logger.add_scalar("Stage {:d}/Learning rate".format(stage), scheduler.get_last_lr()[0], epoch)

                # Auto save whole model between 'checkpoint_interval' epochs
                if checkpoint_history > 0 and epoch % checkpoint_interval == 0:
                    save_checkpoint(settings.CHECKPOINTS_DIR.format(stage=stage), settings.CHECKPOINT_FILE.format(epoch=epoch),
                                    device=device, num_workers=num_workers, val_interval=val_interval, checkpoint_interval=checkpoint_interval,
                                    checkpoint_history=checkpoint_history, init_weights=init_weights, batch_size=batch_size, epochs=epochs,
                                    learning_rate=learning_rate, momentum=momentum, weights_decay=weights_decay, poly_power=poly_power, stage=stage, w1=w1, w2=w2,
                                    description=description, epoch=epoch, ce_train_avg_loss=CE_train_avg_loss.avg, model_state_dict=model.state_dict(),
                                    optimizer_state_dict=optimizer.state_dict())
                    tqdm.write(INFO("Autosaved checkpoint for epoch {0:d} under '{1:s}'.".format(epoch,
                                                                                                 settings.CHECKPOINTS_DIR.format(stage=stage))))

                    # Delete old autosaves, if any
                    checkpoint_epoch_to_delete = epoch - checkpoint_history * checkpoint_interval
                    checkpoint_to_delete_filename = os.path.join(settings.CHECKPOINTS_DIR.format(stage=stage),
                                                                 settings.CHECKPOINT_FILE.format(epoch=checkpoint_epoch_to_delete))
                    if os.path.isfile(checkpoint_to_delete_filename):
                        os.remove(checkpoint_to_delete_filename)

                if epoch % val_interval == 0:
                    # Do validation at epoch intervals of 'val_interval'
                    CE_val_avg_loss, \
                    MSE_val_avg_loss, \
                    FA_val_avg_loss, \
                    Avg_val_loss = do_train_val(do_train=False,
                                                model=model,
                                                device=target_device,
                                                batch_size=batch_size,
                                                stage=stage,
                                                data_loader=val_loader)

                    # Log validation losses for this epoch to TensorBoard
                    val_logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_val_avg_loss.avg, epoch)
                    if stage > 1:
                        val_logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_val_avg_loss.avg, epoch)
                        if stage > 2:
                            val_logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_val_avg_loss.avg, epoch)
                        val_logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_val_loss.avg, epoch)

                # Calculate new learning rate for next epoch
                scheduler.step()

            # Save training weights for this stage
            save_weights(settings.WEIGHTS_DIR.format(stage=stage), settings.FINAL_WEIGHTS_FILE, model)

            process_end_timestamp = datetime.now()
            process_time_taken_hrs = (process_end_timestamp - process_start_timestamp).total_seconds() / timedelta(hours=1).total_seconds()
            train_logger.add_text("INFO",
                                  "Training completed and final weights saved in {0:.2f} hrs at {1:s}.".format(process_time_taken_hrs,
                                                                                                               process_end_timestamp.strftime("%c")),
                                  epochs)
            log_string = "\n################################# Stage {:d} training ENDED #################################".format(stage)
            tqdm.write(log_string)

    elif command == 'test':
        # Testing on a single input image using given weights

        # Create model and set to evaluation mode disabling all batch normalization layers
        model = DSRLSS(stage=1).eval()

        # Load specified weights file
        model.load_state_dict(load_checkpoint_or_weights(weights)['model_state_dict'], strict=True)

        # Copy the model into 'target_device'
        model = model.to(target_device)

        # Load image file, rotate according to EXIF info, add 'batch' dimension and convert to tensor
        with ImageOps.exif_transpose(Image.open(image_file))\
                .convert('RGB')\
                .resize(swapTupleValues(DSRLSS.MODEL_OUTPUT_SIZE), resample=Image.BILINEAR) as input_image:
            with t.no_grad():
                input_transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                         tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD),
                                                         tv.transforms.Resize(size=DSRLSS.MODEL_INPUT_SIZE, interpolation=Image.BILINEAR),
                                                         tv.transforms.Lambda(lambda x: t.unsqueeze(x, dim=0))])
                SSSR_output, _, _, _ = model.forward(input_transform(input_image).to(target_device))
                SSSR_output = np.squeeze(SSSR_output.detach().cpu().numpy(), axis=0)    # Bring back result to CPU memory and remove batch dimension

            # Prepare output image consisting of model input and segmentation image side-by-side (hence '* 2')
            output_image = np.empty((DSRLSS.MODEL_OUTPUT_SIZE[0], DSRLSS.MODEL_OUTPUT_SIZE[1] * 2, consts.NUM_RGB_CHANNELS), dtype=np.uint8)
            argmax_map = np.argmax(SSSR_output, axis=0)

            for y in range(DSRLSS.MODEL_OUTPUT_SIZE[0]):
                for x in range(DSRLSS.MODEL_OUTPUT_SIZE[1]):
                    output_image[y, x, :] = input_image.getpixel((x, y))
                    output_image[y, x + DSRLSS.MODEL_OUTPUT_SIZE[1], :] = cityscapes_settings.CLASS_RGB_COLOR[(argmax_map[y, x])]

        with Image.fromarray(output_image, mode='RGB') as output_image:    # Convert from numpy array to PIL Image
            # Save and show output on plot
            os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
            output_image_filename = os.path.join(settings.OUTPUTS_DIR, os.path.splitext(os.path.basename(image_file))[0] + '.png')

            output_image.save(output_image_filename, format='PNG')
            output_image.show(title='Segmentation output')

        process_end_timestamp = datetime.now()
        process_time_taken_secs = (process_end_timestamp - process_start_timestamp).total_seconds()
        tqdm.write(INFO("Output image saved as: {0:s}. Evaluation required {1:.2f} secs.".format(output_image_filename, process_time_taken_secs)))

    elif command == 'purne_weights':
        with t.no_grad():
            # Purne weights not needed for inference
            model = DSRLSS(stage=1).eval()

            # Load source weights file
            model.load_state_dict(load_checkpoint_or_weights(src_weights)['model_state_dict'], strict=True)

            save_weights(*os.path.split(dest_weights), model)
            tqdm.write(INFO("Output weight saved in '{:s}'.".format(dest_weights)))

    elif command == 'inspect_checkpoint':
        checkpoint_dict = load_checkpoint_or_weights(checkpoint)

        def prettyInnerDictToStr(key, inner_dict):
            if key == 'model_state_dict':   # We only print keys for this dict as it is too big to print
                ret = '{{{0:s}}}'.format(', '.join(inner_dict.keys()))
            else:
                ret = str(inner_dict)

            return '\n{0:s}'.format(ret)

        tqdm.write('\n')
        for key in checkpoint_dict:
            tqdm.write("{0:s}: {1}".format(key,
                                           str(checkpoint_dict[key]) if not isinstance(checkpoint_dict[key], dict) else\
                                               prettyInnerDictToStr(key, checkpoint_dict[key])))

    elif command == 'benchmark':
        # Run benchmark using specified weights and display results

        # Create model and set to evaluation mode disabling all batch normalization layers
        model = DSRLSS(stage=1).eval()

        # Load specified weights file
        model.load_state_dict(load_checkpoint_or_weights(weights)['model_state_dict'], strict=True)

        # Copy the model into 'target_device'
        model = model.to(target_device)

        # Prepare data from CityScapes dataset
        os.makedirs(settings.CITYSCAPES_DATASET_DATA_DIR, exist_ok=True)
        if os.path.getsize(settings.CITYSCAPES_DATASET_DATA_DIR) == 0:
            raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'. Please refer to 'README.md'.".format(settings.CITYSCAPES_DATASET_DATA_DIR)))

        with t.no_grad():
            test_joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                                  lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                                  lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRLSS.MODEL_INPUT_SIZE)(img), seg)])
            test_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                                  split=dataset_split,
                                                  mode='fine',
                                                  target_type='semantic',
                                                  transforms=test_joint_transforms)
            test_loader = t.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # Run benchmark
            miou = mIoU(num_classes=cityscapes_settings.DATASET_NUM_CLASSES)
            for ((input_scaled, _), target) in test_loader:
                SSSR_output, _, _, _ = model.forward(input_scaled.to(target_device))

                # Prepare pred and target for metrices to process
                SSSR_output = SSSR_output.detach().cpu().numpy()    # Bring back result to CPU memory
                target = target.detach().cpu().numpy()

                pred = np.argmax(SSSR_output, axis=1)               # Convert probabilities across dimensions to class label in one 2-D grid

                # Calculate mIoU for this batch
                miou.update(pred, target)

        total_miou = miou()
        tqdm.write(INFO("Computed mIoU: {0:.2f}".format(total_miou)))

        # Save benchmark result to output directories in 'benchmark.txt'
        os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
        output_benchmark_filename = os.path.join(settings.OUTPUTS_DIR, 'benchmark.txt')
        with open(output_benchmark_filename, 'w') as benchmark_file:
            benchmark_file.write("Benchmarking results on Cityscapes dataset's {:s} split\n\n".format(dataset_split))
            benchmark_file.write("On: {:s}\n".format(process_start_timestamp.strftime("%c")))
            benchmark_file.write("Weights file: {:s}\n\n".format(weights))
            benchmark_file.write("mIoU: {:.3f}".format(total_miou))



if __name__ == '__main__':
    assert check_version(sys.version_info, *settings.MIN_PYTHON_VERSION), \
        FATAL("This program needs at least Python {0:d}.{1:d} interpreter.".format(*settings.MIN_PYTHON_VERSION))
    assert check_version(t.__version__, *settings.MIN_PYTORCH_VERSION), \
        FATAL("This program needs at least PyTorch {0:d}.{1:d}.".format(*settings.MIN_PYTORCH_VERSION))

    try:
        parser = argparse.ArgumentParser(description="Implementation of 'Dual Super Resolution Learning For Semantic Segmentation', CVPR 2020 paper.")
        command_parser = parser.add_subparsers(title='commands', dest='command', required=True)

        # Training arguments
        train_parser = command_parser.add_parser('train', help="Train model for different stages")
        train_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        train_parser.add_argument('--num_workers', default=4, type=int, help="No. of workers for data loader")
        train_parser.add_argument('--val_interval', default=10, type=int, help="Epoch intervals after which to perform validation")
        train_parser.add_argument('--checkpoint_interval', default=5, type=int, help="Epoch intervals to create checkpoint after in training")
        train_parser.add_argument('--checkpoint_history', default=5, type=int, help="No. of latest autosaved checkpoints to keep while deleting old ones, 0 to disable autosave")
        train_parser.add_argument('--init_weights', default=None, type=str, help="Load initial weights file for model")
        train_parser.add_argument('--batch_size', default=6, type=int, help="Batch size to use for training and testing")
        train_parser.add_argument('--epochs', required=True, type=int, help="No. of epochs to train")
        train_parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
        train_parser.add_argument('--momentum', type=float, default=0.9, help="Momentum value for SGD")
        train_parser.add_argument('--weights_decay', type=float, default=0.0005, help="Weights decay for SGD")
        train_parser.add_argument('--poly_power', type=float, default=0.9, help="Power for poly learning rate strategy")
        train_parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help="0: Train SSSR only\n1: Train SSSR+SISR\n2: Train SSSR+SISR with feature affinity")
        train_parser.add_argument('--w1', type=float, default=0.1, help="Weight for MSE loss")
        train_parser.add_argument('--w2', type=float, default=1.0, help="Weight for FA loss")
        train_parser.add_argument('--description', type=str, default=None, help="Description of experiment to be saved in 'params.txt' with given commandline parameters")

        # Resume training from checkpoint arguments
        resume_train_parser = command_parser.add_parser('resume_train', help="Resume training model from checkpoint file")
        resume_train_parser.add_argument('--checkpoint', required=True, type=str, help="Resume training with given checkpoint file")

        # Evaluation arguments
        test_parser = command_parser.add_parser('test', help="Test trained weights with a single input image")
        test_parser.add_argument('--image_file', type=str, required=True, help="Run evaluation on a image file using trained weights")
        test_parser.add_argument('--weights', type=str, required=True, help="Weights file to use")
        test_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")

        # Purne weights arguments
        purne_weights_parser = command_parser.add_parser('purne_weights', help="Removes all weights from a weights file which are not needed for inference")
        purne_weights_parser.add_argument('--src_weights', type=str, required=True, help="Checkpoint/Weights file to prune")
        purne_weights_parser.add_argument('--dest_weights', type=str, required=True, help="New weights file to write to")

        # Inspect checkpoint arguments
        inspect_checkpoint_parser = command_parser.add_parser('inspect_checkpoint', help="View contents of a checkpoint file")
        inspect_checkpoint_parser.add_argument('--checkpoint', required=True, type=str, help="Checkpoint file to view contents of")

        # Benchmark arguments
        benchmark_parser = command_parser.add_parser('benchmark', help="Benchmarks model weights to produce metric results")
        benchmark_parser.add_argument('--weights', type=str, required=True, help="Weights to use")
        benchmark_parser.add_argument('--dataset_split', type=str.lower, choices=['train', 'test', 'val'], default='test', help="Which dataset's split to benchmark")
        benchmark_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        benchmark_parser.add_argument('--num_workers', default=4, type=int, help="Number of workers for data loader")
        benchmark_parser.add_argument('--batch_size', default=6, type=int, help="Batch size to use for benchmarking")

        args = parser.parse_args()


        # Validate arguments according to mode
        if args.command == 'train':
            if not args.device in ['cpu', 'gpu'] and not args.device.startswith('cuda'):
                raise argsparse.ArgumentTypeError("'--device' specified must be 'cpu' or 'gpu' or 'cuda:<Device_Index>'!")

            if not args.num_workers >= 0:
                raise argparse.ArgumentTypeError("'--num_workers' should be greater than or equal to 0!")

            if not args.val_interval > 0:
                raise argparse.ArgumentTypeError("'--val_interval' should be greater than 0!")

            if not args.checkpoint_interval > 0:
                raise argparse.ArgumentTypeError("'--checkpoint_interval' should be greater than 0!")

            if not args.checkpoint_history >= 0:
                raise argparse.ArgumentTypeError("'--checkpoint_history' should be greater than or equal (to disable) 0!")

            if args.init_weights:
                if not any(hasExtension(args.init_weights, x) for x in ['.checkpoint', '.weights']):
                    raise argparse.ArgumentTypeError("'--init_weights' must be of either '.checkpoint' or '.weights' file type!")

                if not os.path.isfile(args.init_weights):
                   raise argparse.ArgumentTypeError("Couldn't find initial weights file '{0:s}'!".format(args.init_weights))

            if not args.batch_size > 0:
                raise argparse.ArgumentTypeError("'--batch_size' should be greater than 0!")

            if not args.epochs > 0:
                raise argparse.ArgumentTypeError("'--epochs' should be specified and it must be greater than 0!")

            if not args.learning_rate > 0.:
                raise argparse.ArgumentTypeError("'--learning_rate' should be greater than 0!")

            if not args.momentum > 0.:
                raise argparse.ArgumentTypeError("'--momentum' should be greater than 0!")

            if not args.weights_decay > 0.:
                raise argparse.ArgumentTypeError("'--weights_decay' should be greater than 0!")

            if not args.poly_power > 0.:
                raise argparse.ArgumentTypeError("'--poly_power' should be greater than 0!")

            if args.stage > 1:
                prev_weights_file = os.path.join(settings.WEIGHTS_DIR.format(stage=args.stage-1), settings.FINAL_WEIGHTS_FILE)
                if not os.path.isfile(prev_weights_file):
                    raise argparse.ArgumentTypeError("Couldn't find weights file '{0:s}' from previous stage {1:d}!".format(prev_weights_file, args.stage-1))

            # Warning if there are already weights for this stage
            if os.path.isfile(os.path.join(settings.WEIGHTS_DIR.format(stage=args.stage), settings.FINAL_WEIGHTS_FILE)):
                answer = input(CAUTION("Weights file for this stage already exists. Training will delete the current weights and logs. Continue? (y/n) ")).lower()
                if answer == 'y':
                    shutil.rmtree(settings.LOGS_DIR.format(stage=args.stage, mode=''), ignore_errors=True)
                    shutil.rmtree(settings.WEIGHTS_DIR.format(stage=args.stage))
                else:
                    sys.exit(0)

        elif args.command == 'resume_train':
            if not hasExtension(args.checkpoint, '.checkpoint'):
                raise argparse.ArgumentTypeError("Please specify a '.checkpoint' file as the whole model and optimizer states needs to be loaded!")

            if not os.path.isfile(args.checkpoint):
                raise argparse.ArgumentTypeError("Couldn't find checkpoint file '{0:s}'!".format(args.checkpoint))

        elif args.command == 'test':
            if not os.path.isfile(args.image_file):
                raise argparse.ArgumentTypeError("File specified in '--image_file' parameter doesn't exists!")

            if not any(hasExtension(args.weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.weights):
                raise argparse.ArgumentTypeError("Couldn't find weights file '{:s}'!".format(args.weights))

            if not args.device in ['cpu', 'gpu'] and not args.device.startswith('cuda'):
                raise argsparse.ArgumentTypeError("'--device' specified must be 'cpu' or 'gpu' or 'cuda:<Device_Index>'!")

        elif args.command == 'purne_weights':
            if not any(hasExtension(args.src_weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--src_weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.src_weights):
                raise argparse.ArgumentTypeError("File specified in '--src_weights' parameter doesn't exists!")

            if os.path.isfile(args.dest_weights):
                answer = input(CAUTION("Destination weights file specified already exists. This will overwrite the file. Continue (y/n)? ")).lower()
                if answer != 'y':
                    sys.exit(0)

        elif args.command == 'inspect_checkpoint':
            if not hasExtension(args.checkpoint, '.checkpoint'):
                raise argparse.ArgumentTypeError("Please specify a '.checkpoint' file!")

            if not os.path.isfile(args.checkpoint):
                raise argparse.ArgumentTypeError("Couldn't find checkpoint file '{0:s}'!".format(args.checkpoint))

        elif args.command == 'benchmark':
            if not any(hasExtension(args.weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.weights):
                raise argparse.ArgumentTypeError("Couldn't find the specified weights file '{:s}'!".format(args.weights))

            if not args.device in ['cpu', 'gpu'] and not args.device.startswith('cuda'):
                raise argsparse.ArgumentTypeError("'--device' specified must be 'cpu' or 'gpu' or 'cuda:<Device_Index>'!")

            if not args.num_workers >= 0:
                raise argparse.ArgumentTypeError("'--num_workers' should be greater than or equal to 0!")

            if not args.batch_size > 0:
                raise argparse.ArgumentTypeError("'--batch_size' should be greater than 0!")

        # Do action in 'command'
        assert args.command in ['train', 'resume_train', 'test', 'purne_weights', 'inspect_checkpoint', 'benchmark'],\
            "BUG CHECK: Unimplemented 'args.command': {:s}!".format(args.command)
        main(**args.__dict__)


    except KeyboardInterrupt:
        tqdm.write(INFO("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))

    except argparse.ArgumentTypeError as ex:
        tqdm.write(FATAL("{:s}".format(str(ex))))
        tqdm.write('\n')
        parser.print_usage()