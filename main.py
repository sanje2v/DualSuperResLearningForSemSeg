import sys
import os
import os.path
import shutil
import argparse
import termcolor
from tqdm.auto import tqdm as tqdm
from pydoc import locate as str2type
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
from metrices import *
from utils import *
import consts
import settings
from datasets.Cityscapes import settings as cityscapes_settings


# GLOBALS
g_profiler = None


def do_train_val(do_train: bool,
                 model,
                 target_device,
                 batch_size,
                 stage,
                 data_loader,
                 w1,
                 w2,
                 freeze_batch_norm=False,
                 optimizer=None,
                 scheduler=None):
    # Set model to either training or testing mode
    model.train(mode=do_train)

    # If training and freeze BatchNorm layer option is ON, then freeze them
    if do_train and freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, t.nn.BatchNorm2d):
                module.eval()

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

            input_scaled = input_scaled.to(target_device)
            input_org = (None if stage == 1 else input_org.to(target_device))
            target = target.to(target_device)
            if do_train:
                optimizer.zero_grad()

            SSSR_output, SISR_output, SSSR_transform_output, SISR_transform_output = model.forward(input_scaled)
            # SANITY CHECK: Check network outputs doesn't have any 'NaN' values
            assert not (t.isnan(SSSR_output).any().item()),\
                FATAL("SSSR network output contains 'NaN' values and so cannot continue. Exiting.")
            assert not (False if SISR_output is None else t.isnan(SISR_output).any().item()),\
                FATAL("SISSR network output contains 'NaN' values and so cannot continue. Exiting.")

            CE_loss = t.nn.CrossEntropyLoss(ignore_index=cityscapes_settings.IGNORE_CLASS_LABEL)(SSSR_output, target)
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


def main(command,
         device=None,
         disable_cudnn_benchmark=None,
         profile=None,
         num_workers=None,
         val_interval=None,
         checkpoint=None,
         checkpoint_interval=None,
         checkpoint_history=None,
         init_weights=None,
         batch_size=None,
         epochs=None,
         learning_rate=None,
         end_learning_rate=None,
         momentum=None,
         weights_decay=None,
         poly_power=None,
         stage=None,
         w1=None,
         w2=None,
         description=None,
         freeze_batch_norm=None,
         image_file=None,
         weights=None,
         src_weights=None,
         dest_weights=None,
         dataset_split=None,
         key=None,
         value=None,
         typeof=None):

    # Time keeper
    process_start_timestamp = datetime.now()

    if command == 'resume-train':
        checkpoint_dict = load_checkpoint_or_weights(checkpoint)

        device = checkpoint_dict['device']
        disable_cudnn_benchmark = checkpoint_dict['disable_cudnn_benchmark']
        num_workers = checkpoint_dict['num_workers']
        val_interval = checkpoint_dict['val_interval']
        checkpoint_interval = checkpoint_dict['checkpoint_interval']
        checkpoint_history = checkpoint_dict['checkpoint_history']
        init_weights = checkpoint_dict['init_weights']
        batch_size = checkpoint_dict['batch_size']
        epochs = checkpoint_dict['epochs']
        learning_rate = checkpoint_dict['learning_rate']
        end_learning_rate = checkpoint_dict['end_learning_rate']
        momentum = checkpoint_dict['momentum']
        weights_decay = checkpoint_dict['weights_decay']
        poly_power = checkpoint_dict['poly_power']
        stage = checkpoint_dict['stage']
        w1 = checkpoint_dict['w1']
        w2 = checkpoint_dict['w2']
        freeze_batch_norm = checkpoint_dict['freeze_batch_norm']
        description = checkpoint_dict['description']
        best_validation_dict = checkpoint_dict['best_validation_dict']

    if device:
        # Device to perform calculation in
        if isCUDAdevice(device):
           if not t.cuda.is_available():
               raise Exception("CUDA is not available to use for accelerated computing!")

           if disable_cudnn_benchmark is not None:
               t.backends.cudnn.benchmark = not disable_cudnn_benchmark
        target_device = t.device('cuda' if device == 'gpu' else device)

    if command in ['train', 'resume-train']:
        # Training and Validation on dataset mode

        if command == 'train':
            best_validation_dict = {}

        # Prevent system from entering sleep state so that long training session is not interrupted
        if prevent_system_sleep():
            tqdm.write(INFO("System will NOT be allowed to sleep until this training is complete/interrupted."))
        else:
            tqdm.write(CAUTION("Please make sure system is NOT configured to sleep on idle! Sleep mode will pause training."))

        # Create model according to stage
        model = DSRLSS(stage)

        if command == 'resume-train':
            model.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
        else:
            # Load initial weight, if any
            if init_weights:
                model.load_state_dict(load_checkpoint_or_weights(init_weights, map_location=target_device)['model_state_dict'], strict=False)
            else:
                # Load checkpoint from previous stage, if not the first stage
                if stage == 1:
                    tqdm.write(INFO("Pretrained weights for ResNet101 will be used to initialize network before training."))
                    model.initialize_with_pretrained_weights(settings.WEIGHTS_ROOT_DIR, map_location=target_device)
                else:
                    prev_weights_filename = os.path.join(settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE)
                    if os.path.isfile(prev_weights_filename):
                        tqdm.write(INFO("'{0:s}' weights file from previous stage was found and will be used to initialize network before training.".format(prev_weights_filename)))
                        weights_dict = load_checkpoint_or_weights(os.path.join(settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE), map_location=target_device)
                        model.load_state_dict(weights_dict['model_state_dict'], strict=False)
                    else:
                        tqdm.write(CAUTION("'{0:s}' weights file from previous stage was not found and network weights were initialized with Pytorch's default method.".format(prev_weights_filename)))

        # Copy the model into 'target_device' memory
        model = model.to(target_device)

        # Print number of training parameters
        tqdm.write(INFO("Total training parameters: {:,}".format(countNoOfModelParams(model, only_training_params=True))))

        # Prepare data from CityScapes dataset
        os.makedirs(settings.CITYSCAPES_DATASET_DATA_DIR, exist_ok=True)
        if os.path.getsize(settings.CITYSCAPES_DATASET_DATA_DIR) == 0:
            raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'.".format(settings.CITYSCAPES_DATASET_DATA_DIR)))

        train_joint_transforms = JointCompose([JointRandomRotate(degrees=15.0, fill=(0, 0)),
                                               JointRandomCrop(min_scale=1.0, max_scale=3.5),
                                               JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                               lambda img, seg: (ColorJitter2(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)(img), seg),
                                               JointHFlip(),
                                               # CAUTION: 'kernel_size' should be > 0 and odd integer
                                               lambda img, seg: (tv.transforms.RandomApply([tv.transforms.GaussianBlur(kernel_size=3)], p=0.5)(img), seg),
                                               lambda img, seg: (tv.transforms.RandomGrayscale(p=0.1)(img), seg),
                                               lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                               lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRLSS.MODEL_INPUT_SIZE)(img), seg)])
        train_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                               split='train',
                                               mode='fine',
                                               target_type='semantic',
                                               transforms=train_joint_transforms)
        train_loader = t.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=isCUDAdevice(device),
                                               drop_last=True)

        val_joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                             lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                             lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRLSS.MODEL_INPUT_SIZE)(img), seg)])
        val_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                             split='val',
                                             mode='fine',
                                             target_type='semantic',
                                             transforms=val_joint_transforms)
        val_loader = t.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=isCUDAdevice(device),
                                             drop_last=False)

        # Make sure proper log directories exist
        train_logs_dir = settings.LOGS_DIR.format(stage=stage, mode='train')
        val_logs_dir = settings.LOGS_DIR.format(stage=stage, mode='val')
        os.makedirs(train_logs_dir, exist_ok=True)
        os.makedirs(val_logs_dir, exist_ok=True)

        # Write training parameters provided to params.txt log file
        write_params_file(os.path.join(train_logs_dir, settings.PARAMS_FILE),
                          "Timestamp: {:s}".format(process_start_timestamp.strftime("%c")),
                          "Device: {:s}".format(device),
                          "Disable CUDNN benchmark mode: {:}".format(disable_cudnn_benchmark) if isCUDAdevice(device) else None,
                          "No. of workers: {:d}".format(num_workers),
                          "Validation interval: {:d}".format(val_interval),
                          "Checkpoint interval: {:d}".format(checkpoint_interval),
                          "Checkpoint history: {:d}".format(checkpoint_history),
                          "Initial weights: {:s}".format(init_weights) if init_weights else None,
                          "Resume checkpoint: {:s}".format(checkpoint) if checkpoint else None,
                          "Batch size: {:d}".format(batch_size),
                          "Epochs: {:d}".format(epochs),
                          "Learning rate: {:f}".format(learning_rate),
                          "End learning rate: {:f}".format(end_learning_rate),
                          "Momentum: {:f}".format(momentum),
                          "Weights decay: {:f}".format(weights_decay),
                          "Poly power: {:f}".format(poly_power),
                          "Stage: {:d}".format(stage),
                          "Loss Weight 1: {:.4f}".format(w1) if stage > 1 else None,
                          "Loss Weight 2: {:.4f}".format(w2) if stage > 2 else None,
                          "Freeze batch normalization: {:}".format(freeze_batch_norm),
                          "Description: {:s}".format(description) if description else None)

        # Start training and validation
        with tb.SummaryWriter(log_dir=train_logs_dir) as train_logger,\
             tb.SummaryWriter(log_dir=val_logs_dir) as val_logger:

            # Training optimizer and schedular
            optimizer = t.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weights_decay)
            if command == 'resume-train':
                optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                starting_epoch = checkpoint_dict['epoch']
            else:
                starting_epoch = 0

            scheduler = PolynomialLR(optimizer,
                                     max_decay_steps=epochs,
                                     end_learning_rate=end_learning_rate,
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
                training_epoch_begin_timestamp = datetime.now()
                CE_train_avg_loss, \
                MSE_train_avg_loss, \
                FA_train_avg_loss, \
                Avg_train_loss = do_train_val(do_train=True,
                                              model=model,
                                              target_device=target_device,
                                              batch_size=batch_size,
                                              stage=stage,
                                              data_loader=train_loader,
                                              w1=w1,
                                              w2=w2,
                                              freeze_batch_norm=freeze_batch_norm,
                                              optimizer=optimizer,
                                              scheduler=scheduler)

                # Print estimated time for training completion
                training_epoch_timetaken = (datetime.now() - training_epoch_begin_timestamp).total_seconds() / timedelta(hours=1).total_seconds()
                tqdm.write("Training to complete in: {:.2f} hr(s)".format(training_epoch_timetaken * (epochs - epoch)))

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
                                    description=description, freeze_batch_norm=freeze_batch_norm, epoch=epoch, best_validation_dict=best_validation_dict,
                                    ce_train_avg_loss=CE_train_avg_loss.avg, model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict())
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
                                                target_device=target_device,
                                                batch_size=batch_size,
                                                stage=stage,
                                                data_loader=val_loader,
                                                w1=w1,
                                                w2=w2)

                    # Save epoch number and total error of best validation and then checkpoint
                    if not best_validation_dict or Avg_val_loss.avg < best_validation_dict['loss']:
                        best_validation_dict['epoch'] = epoch
                        best_validation_dict['loss'] = Avg_val_loss.avg

                        save_checkpoint(settings.CHECKPOINTS_DIR.format(stage=stage), settings.CHECKPOINT_FILE.format(epoch='_bestval'),
                                        device=device, num_workers=num_workers, val_interval=val_interval, checkpoint_interval=checkpoint_interval,
                                        checkpoint_history=checkpoint_history, init_weights=init_weights, batch_size=batch_size, epochs=epochs,
                                        learning_rate=learning_rate, momentum=momentum, weights_decay=weights_decay, poly_power=poly_power, stage=stage, w1=w1, w2=w2,
                                        description=description, epoch=epoch, best_validation_dict=best_validation_dict, ce_train_avg_loss=CE_train_avg_loss.avg,
                                        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict())

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
        model.load_state_dict(load_checkpoint_or_weights(weights, map_location=target_device)['model_state_dict'], strict=True)

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
            output_image = np.zeros((DSRLSS.MODEL_OUTPUT_SIZE[0], DSRLSS.MODEL_OUTPUT_SIZE[1] * 2, consts.NUM_RGB_CHANNELS), dtype=np.uint8)
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

    elif command == 'print-model':
        model = DSRLSS(stage)
        tqdm.write(str(model))
        info = "\nTotal training parameters: {0:,}\nTotal parameters: {1:,}".format(countNoOfModelParams(model, only_training_params=True),
                                                                                    countNoOfModelParams(model, only_training_params=False))
        tqdm.write(INFO(info))

    elif command == 'purne-weights':
        with t.no_grad():
            # Purne weights not needed for inference
            model = DSRLSS(stage=1).eval()

            # Load source weights file
            model.load_state_dict(load_checkpoint_or_weights(src_weights)['model_state_dict'], strict=True)

            save_weights(*os.path.split(dest_weights), model)
            tqdm.write(INFO("Output weight saved in '{:s}'.".format(dest_weights)))

    elif command == 'inspect-checkpoint':
        checkpoint_dict = load_checkpoint_or_weights(checkpoint)

        def prettyDictToStr(dict_):
            output = []
            for key in dict_:
                if isinstance(dict_[key], dict):
                    output.append("{0}: {1}".format(key, prettyDictToStr(dict_[key])))
                elif isinstance(dict_[key], (np.ndarray, t.Tensor, list)):
                    output.append(key)
                else:
                    if isinstance(dict_[key], str):
                        output.append("{0:s}: \"{1}\"".format(key, str(dict_[key])))
                    else:
                        output.append("{0:s}: {1}".format(key, str(dict_[key])))

            return "{{{:s}}}".format(', '.join(output))

        tqdm.write(prettyDictToStr(checkpoint_dict))

    elif command == 'edit-checkpoint':
        checkpoint_dict = load_checkpoint_or_weights(checkpoint)
        checkpoint_dict[key] = str2type(typeof)(value)
        save_checkpoint(*os.path.split(checkpoint), **checkpoint_dict)

    elif command == 'benchmark':
        # Run benchmark using specified weights and display results

        # Create model and set to evaluation mode disabling all batch normalization layers
        model = DSRLSS(stage=1).eval()

        # Load specified weights file
        model.load_state_dict(load_checkpoint_or_weights(weights, map_location=target_device)['model_state_dict'], strict=True)

        # Copy the model into 'target_device'
        model = model.to(target_device)

        # Prepare data from CityScapes dataset
        os.makedirs(settings.CITYSCAPES_DATASET_DATA_DIR, exist_ok=True)
        if os.path.getsize(settings.CITYSCAPES_DATASET_DATA_DIR) == 0:
            raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'. Please refer to 'README.md'.".format(settings.CITYSCAPES_DATASET_DATA_DIR)))

        test_joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                              lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                              lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRLSS.MODEL_INPUT_SIZE)(img), seg)])
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
                SSSR_output, _, _, _ = model.forward(input_scaled.to(target_device))
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



if __name__ == '__main__':
    assert check_version(sys.version_info, *settings.MIN_PYTHON_VERSION), \
        FATAL("This program needs at least Python {0:d}.{1:d} interpreter.".format(*settings.MIN_PYTHON_VERSION))
    assert check_version(t.__version__, *settings.MIN_PYTORCH_VERSION), \
        FATAL("This program needs at least PyTorch {0:d}.{1:d}.".format(*settings.MIN_PYTORCH_VERSION))
    assert check_version(tv.__version__, *settings.MIN_TORCHVISION_VERSION), \
        FATAL("This program needs at least TorchVision {0:d}.{1:d}.".format(*settings.MIN_TORCHVISION_VERSION))
    assert check_version(np.__version__, *settings.MIN_NUMPY_VERSION), \
        FATAL("This program needs at least NumPy {0:d}.{1:d}.".format(*settings.MIN_PYTORCH_VERSION))

    do_profiling = False
    try:
        parser = argparse.ArgumentParser(description="Implementation of 'Dual Super Resolution Learning For Semantic Segmentation', CVPR 2020 paper.")
        command_parser = parser.add_subparsers(title='commands', dest='command', required=True)

        # Training arguments
        train_parser = command_parser.add_parser('train', help="Train model for different stages")
        train_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        train_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make training slower")
        train_parser.add_argument('--profile', action='store_true', help="Enable PyTorch profiling of execution times and memory usage")
        train_parser.add_argument('--num-workers', default=4, type=int, help="No. of workers for data loader")
        train_parser.add_argument('--val-interval', default=10, type=int, help="Epoch intervals after which to perform validation")
        train_parser.add_argument('--checkpoint-interval', default=5, type=int, help="Epoch intervals to create checkpoint after in training")
        train_parser.add_argument('--checkpoint-history', default=5, type=int, help="No. of latest autosaved checkpoints to keep while deleting old ones, 0 to disable autosave")
        train_parser.add_argument('--init-weights', default=None, type=str, help="Load initial weights file for model")
        train_parser.add_argument('--batch-size', default=6, type=int, help="Batch size to use for training and testing")
        train_parser.add_argument('--epochs', required=True, type=int, help="No. of epochs to train")
        train_parser.add_argument('--learning-rate', type=float, default=0.01, help="Learning rate to begin training with")
        train_parser.add_argument('--end-learning-rate', type=float, default=0.001, help="End learning rate for the last epoch")
        train_parser.add_argument('--momentum', type=float, default=0.9, help="Momentum value for SGD")
        train_parser.add_argument('--weights-decay', type=float, default=0.0005, help="Weights decay for SGD")
        train_parser.add_argument('--poly-power', type=float, default=0.9, help="Power for poly learning rate strategy")
        train_parser.add_argument('--stage', type=int, choices=settings.STAGES, required=True, help="0: Train SSSR only\n1: Train SSSR+SISR\n2: Train SSSR+SISR with feature affinity")
        train_parser.add_argument('--w1', type=float, default=0.1, help="Weight for MSE loss")
        train_parser.add_argument('--w2', type=float, default=1.0, help="Weight for FA loss")
        train_parser.add_argument('--freeze-batch-norm', action='store_true', help="Keep all Batch Normalization layers disabled while training")
        train_parser.add_argument('--description', type=str, default=None, help="Description of experiment to be saved in 'params.txt' with given commandline parameters")

        # Resume training from checkpoint arguments
        resume_train_parser = command_parser.add_parser('resume-train', help="Resume training model from checkpoint file")
        resume_train_parser.add_argument('--checkpoint', required=True, type=str, help="Resume training with given checkpoint file")

        # Evaluation arguments
        test_parser = command_parser.add_parser('test', help="Test trained weights with a single input image")
        test_parser.add_argument('--image-file', type=str, required=True, help="Run evaluation on a image file using trained weights")
        test_parser.add_argument('--weights', type=str, required=True, help="Weights file to use")
        test_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        test_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make evaluation slower")
        test_parser.add_argument('--profile', action='store_true', help="Enable PyTorch profiling of execution times and memory usage")

        # Print model arguments
        print_model_parser = command_parser.add_parser('print-model', help="Prints all the layers in the model for a stage")
        print_model_parser.add_argument('--stage', type=int, choices=settings.STAGES, help="Stage to print layers of model for")

        # Purne weights arguments
        purne_weights_parser = command_parser.add_parser('purne-weights', help="Removes all weights from a weights file which are not needed for inference")
        purne_weights_parser.add_argument('--src-weights', type=str, required=True, help="Checkpoint/Weights file to prune")
        purne_weights_parser.add_argument('--dest-weights', type=str, required=True, help="New weights file to write to")

        # Inspect checkpoint arguments
        inspect_checkpoint_parser = command_parser.add_parser('inspect-checkpoint', help="View contents of a checkpoint file")
        inspect_checkpoint_parser.add_argument('--checkpoint', required=True, type=str, help="Checkpoint file to view contents of")

        # Edit checkpoint arguments
        edit_checkpoint_parser = command_parser.add_parser('edit-checkpoint', help="Edit contents of a checkpoint file")
        edit_checkpoint_parser.add_argument('--checkpoint', required=True, type=str, help="Checkpoint file to edit contents of")
        edit_checkpoint_parser.add_argument('--key', required=True, type=str, help="Specify key of the dictionary of checkpoint to edit")
        edit_checkpoint_parser.add_argument('--value', required=True, type=str, help="Specify value of the key to edit")
        edit_checkpoint_parser.add_argument('--typeof', required=True, type=str, help="Specify type of the specified value")

        # Benchmark arguments
        benchmark_parser = command_parser.add_parser('benchmark', help="Benchmarks model weights to produce metric results")
        benchmark_parser.add_argument('--weights', type=str, required=True, help="Weights to use")
        benchmark_parser.add_argument('--dataset-split', type=str.lower, choices=['train', 'test', 'val'], default='test', help="Which dataset's split to benchmark")
        benchmark_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        benchmark_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make training slower")
        benchmark_parser.add_argument('--num-workers', default=4, type=int, help="Number of workers for data loader")
        benchmark_parser.add_argument('--batch-size', default=6, type=int, help="Batch size to use for benchmarking")

        args = parser.parse_args()


        # Validate arguments according to mode
        if args.command == 'train':
            if not args.device in ['cpu', 'gpu'] and not args.device.startswith('cuda'):
                raise argparse.ArgumentTypeError("'--device' specified must be 'cpu' or 'gpu' or 'cuda:<Device_Index>'!")

            if not isCUDAdevice(args.device) and args.disable_cudnn_benchmark:
                raise argparse.ArgumentTypeError("'--disable-cudnn-benchmark' is unsupported in non-CUDA devices!")

            if not args.num_workers >= 0:
                raise argparse.ArgumentTypeError("'--num-workers' should be greater than or equal to 0!")

            if not args.val_interval > 0:
                raise argparse.ArgumentTypeError("'--val-interval' should be greater than 0!")

            if not args.checkpoint_interval > 0:
                raise argparse.ArgumentTypeError("'--checkpoint-interval' should be greater than 0!")

            if not args.checkpoint_history >= 0:
                raise argparse.ArgumentTypeError("'--checkpoint-history' should be greater than or equal (to disable) 0!")

            if args.init_weights:
                if not any(hasExtension(args.init_weights, x) for x in ['.checkpoint', '.weights']):
                    raise argparse.ArgumentTypeError("'--init-weights' must be of either '.checkpoint' or '.weights' file type!")

                if not os.path.isfile(args.init_weights):
                   raise argparse.ArgumentTypeError("Couldn't find initial weights file '{0:s}'!".format(args.init_weights))

            if not args.batch_size > 0:
                raise argparse.ArgumentTypeError("'--batch-size' should be greater than 0!")

            if not args.epochs > 0:
                raise argparse.ArgumentTypeError("'--epochs' should be specified and it must be greater than 0!")

            if not args.learning_rate > 0.:
                raise argparse.ArgumentTypeError("'--learning-rate' should be greater than 0!")

            if not args.momentum > 0.:
                raise argparse.ArgumentTypeError("'--momentum' should be greater than 0!")

            if not args.weights_decay > 0.:
                raise argparse.ArgumentTypeError("'--weights-decay' should be greater than 0!")

            if not args.poly_power > 0.:
                raise argparse.ArgumentTypeError("'--poly-power' should be greater than 0!")

            # Warning if there are already weights for this stage
            if os.path.isfile(os.path.join(settings.WEIGHTS_DIR.format(stage=args.stage), settings.FINAL_WEIGHTS_FILE)):
                answer = input(CAUTION("Weights file for this stage already exists. Training will delete the current weights and logs. Continue? (y/n) ")).lower()
                if answer == 'y':
                    shutil.rmtree(settings.LOGS_DIR.format(stage=args.stage, mode=''), ignore_errors=True)
                    shutil.rmtree(settings.WEIGHTS_DIR.format(stage=args.stage))
                else:
                    sys.exit(0)

            # Enable profiler if '--profile' option is specified
            do_profiling = args.profile

        elif args.command == 'resume-train':
            if not hasExtension(args.checkpoint, '.checkpoint'):
                raise argparse.ArgumentTypeError("Please specify a '.checkpoint' file as the whole model and optimizer states needs to be loaded!")

            if not os.path.isfile(args.checkpoint):
                raise argparse.ArgumentTypeError("Couldn't find checkpoint file '{0:s}'!".format(args.checkpoint))

        elif args.command == 'test':
            if not os.path.isfile(args.image_file):
                raise argparse.ArgumentTypeError("File specified in '--image-file' parameter doesn't exists!")

            if not any(hasExtension(args.weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.weights):
                raise argparse.ArgumentTypeError("Couldn't find weights file '{:s}'!".format(args.weights))

            if not args.device in ['cpu', 'gpu'] and not args.device.startswith('cuda'):
                raise argparse.ArgumentTypeError("'--device' specified must be 'cpu' or 'gpu' or 'cuda:<Device_Index>'!")

            if not isCUDAdevice(args.device) and args.disable_cudnn_benchmark:
                raise argparse.ArgumentTypeError("'--disable-cudnn-benchmark' is unsupported in non-CUDA devices!")

            # Enable profiler if '--profile' option is specified
            do_profiling = args.profile

        elif args.command == 'purne-weights':
            if not any(hasExtension(args.src_weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--src-weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.src_weights):
                raise argparse.ArgumentTypeError("File specified in '--src-weights' parameter doesn't exists!")

            if os.path.isfile(args.dest_weights):
                answer = input(CAUTION("Destination weights file specified already exists. This will overwrite the file. Continue (y/n)? ")).lower()
                if answer != 'y':
                    sys.exit(0)

        elif args.command == 'inspect-checkpoint':
            if not hasExtension(args.checkpoint, '.checkpoint'):
                raise argparse.ArgumentTypeError("Please specify a '.checkpoint' file!")

            if not os.path.isfile(args.checkpoint):
                raise argparse.ArgumentTypeError("Couldn't find checkpoint file '{0:s}'!".format(args.checkpoint))

        elif args.command == 'edit-checkpoint':
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
                raise argparse.ArgumentTypeError("'--device' specified must be 'cpu' or 'gpu' or 'cuda:<Device_Index>'!")

            if not isCUDAdevice(args.device) and args.disable_cudnn_benchmark:
                raise argparse.ArgumentTypeError("'--disable-cudnn-benchmark' is unsupported in non-CUDA devices!")

            if not args.num_workers >= 0:
                raise argparse.ArgumentTypeError("'--num-workers' should be greater than or equal to 0!")

            if not args.batch_size > 0:
                raise argparse.ArgumentTypeError("'--batch-size' should be greater than 0!")

        # Do action in 'command'
        assert args.command in ['train', 'resume-train', 'test', 'print-model', 'purne-weights', 'inspect-checkpoint', 'edit-checkpoint', 'benchmark'],\
            "BUG CHECK: Unimplemented 'args.command': {:s}.".format(args.command)

        with t.autograd.profiler.profile(enabled=do_profiling,
                                         use_cuda=hasattr(args, 'device') and isCUDAdevice(args.device),
                                         record_shapes=True,
                                         profile_memory=True) as profiler:
            g_profiler = profiler
            main(**args.__dict__)


    except KeyboardInterrupt:
        tqdm.write(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))

    except argparse.ArgumentTypeError as ex:
        tqdm.write(FATAL("{:s}".format(str(ex))))
        tqdm.write('\n')
        parser.print_usage()

    finally:
        # If a profiler is active, stop it and save results to disk
        if g_profiler:
            profiling_filename = os.path.join(settings.OUTPUTS_DIR, settings.PROFILING_FILE)
            g_profiler.export_chrome_trace(profiling_filename)
            tqdm.write(INFO("Profiling output has been saved to: {:s}.".format(profiling_filename)))