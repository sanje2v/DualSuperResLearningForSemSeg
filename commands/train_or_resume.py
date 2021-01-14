import os
import os.path
from tqdm.auto import tqdm as tqdm
import numpy as np
import torch as t
import torchvision as tv
from torch.utils import tensorboard as tb
from datetime import datetime, timedelta

from models import DSRL
from models.schedulers import PolynomialLR
from models.losses import FALoss
from models.transforms import *
from metrices import *
from utils import *
import settings
from datasets.Cityscapes import settings as cityscapes_settings



def train_or_resume(command, device, disable_cudnn_benchmark, device_obj, num_workers, val_interval, checkpoint_interval,
                    checkpoint_history, init_weights, batch_size, epochs, learning_rate, end_learning_rate, momentum,
                    weights_decay, poly_power, stage, w1, w2, freeze_batch_norm, description, **other_args):
    # Training and Validation on dataset mode
    
    # Time keeper
    process_start_timestamp = datetime.now()

    if command == 'train':
        best_validation_dict = {}

    # Prevent system from entering sleep state so that long training session is not interrupted
    if prevent_system_sleep():
        tqdm.write(INFO("System will NOT be allowed to sleep until this training is complete/interrupted."))
    else:
        tqdm.write(CAUTION("Please make sure system is NOT configured to sleep on idle! Sleep mode will pause training."))

    # Create model according to stage
    model = DSRL(stage)

    if command == 'resume-train':
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
    else:
        # Load initial weight, if any
        if init_weights:
            model.load_state_dict(load_checkpoint_or_weights(init_weights)['model_state_dict'], strict=False)
        else:
            # Load checkpoint from previous stage, if not the first stage
            if stage == 1:
                tqdm.write(INFO("Pretrained weights for ResNet101 will be used to initialize network before training."))
                model.initialize_with_pretrained_weights(settings.WEIGHTS_ROOT_DIR)
            else:
                prev_weights_filename = os.path.join(settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE)
                if os.path.isfile(prev_weights_filename):
                    tqdm.write(INFO("'{0:s}' weights file from previous stage was found and will be used to initialize network before training.".format(prev_weights_filename)))
                    weights_dict = load_checkpoint_or_weights(os.path.join(settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE))
                    model.load_state_dict(weights_dict['model_state_dict'], strict=False)
                else:
                    tqdm.write(CAUTION("'{0:s}' weights file from previous stage was not found and network weights were initialized with Pytorch's default method.".format(prev_weights_filename)))

    # Copy the model into 'device_obj' memory
    # NOTE: We only copy model to specific device after applying weights in CPU.
    model = model.to(device_obj)

    # Print number of training parameters
    tqdm.write(INFO("Total training parameters: {:,}".format(countNoOfModelParams(model)[0])))

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
                                           lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg)])
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
                                         lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg)])
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
    _write_params_file(os.path.join(train_logs_dir, settings.PARAMS_FILE),
                       "Timestamp: {:s}".format(process_start_timestamp.strftime("%c")),
                       "Device: {:s}".format(device),
                       "Disable CUDNN benchmark mode: {:}".format(disable_cudnn_benchmark) if isCUDAdevice(device) else None,
                       "No. of workers: {:d}".format(num_workers),
                       "Validation interval: {:d}".format(val_interval),
                       "Checkpoint interval: {:d}".format(checkpoint_interval),
                       "Checkpoint history: {:d}".format(checkpoint_history),
                       "Initial weights: {:s}".format(init_weights) if init_weights else None,
                       "Resuming checkpoint: {:s}".format(other_args['checkpoint']) if command =='resume-train' else None,
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
                                 last_epoch=(starting_epoch - 1))

        # Start training and then validation after specific intervals
        train_logger.add_text("INFO", "Training started on {:s}".format(process_start_timestamp.strftime("%c")), 1)
        log_string = "################################# Stage {:d} training STARTED #################################".format(stage)
        tqdm.write('\n' + INFO(log_string))

        training_epoch_timetaken_list = []
        for epoch in range((starting_epoch + 1), (epochs + 1)):
            log_string = "\nEPOCH {0:d}/{1:d}".format(epoch, epochs)
            tqdm.write(log_string)

            # Do training for this epoch
            training_epoch_begin_timestamp = datetime.now()
            CE_train_avg_loss, \
            MSE_train_avg_loss, \
            FA_train_avg_loss, \
            Avg_train_loss = _do_train_val(do_train=True,
                                           model=model,
                                           device_obj=device_obj,
                                           batch_size=batch_size,
                                           stage=stage,
                                           data_loader=train_loader,
                                           w1=w1,
                                           w2=w2,
                                           freeze_batch_norm=freeze_batch_norm,
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
                Avg_val_loss = _do_train_val(do_train=False,
                                             model=model,
                                             device_obj=device_obj,
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

            # Print estimated time for training completion
            training_epoch_timetaken_list.append((datetime.now() - training_epoch_begin_timestamp).total_seconds())
            training_epoch_timetaken = np.mean(training_epoch_timetaken_list[(-val_interval*2):])   # NOTE: '*2' due to Nyquist sampling theorem
            tqdm.write("Est. training completion in {:s}.".format(makeSecondsPretty(training_epoch_timetaken * (epochs - epoch))))

        # Save training weights for this stage
        save_weights(settings.WEIGHTS_DIR.format(stage=stage), settings.FINAL_WEIGHTS_FILE, model)

        process_end_timestamp = datetime.now()
        process_time_taken_hrs = (process_end_timestamp - process_start_timestamp).total_seconds() / timedelta(hours=1).total_seconds()
        train_logger.add_text("INFO",
                              "Training took {0:s} and completed on {1:s}.".format(makeSecondsPretty(process_time_taken_hrs),
                                                                                   process_end_timestamp.strftime("%c")),
                              epochs)
        log_string = "################################# Stage {:d} training ENDED #################################".format(stage)
        tqdm.write('\n' + INFO(log_string))


def _do_train_val(do_train,
                  model,
                  device_obj,
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

            input_scaled = input_scaled.to(device_obj)
            input_org = (None if stage == 1 else input_org.to(device_obj))
            target = target.to(device_obj)
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


def _write_params_file(filename, *list_params):
    with open(filename, mode='w') as params_file:
        for params_str in list_params:
            if params_str:
                params_file.write(params_str)
                params_file.write('\n')     # NOTE: '\n' here automatically converts it to newline for the current platform