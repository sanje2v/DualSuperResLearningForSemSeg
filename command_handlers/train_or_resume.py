import gc
import os
import os.path
import termcolor
import numpy as np
import apex
import torch as t
import torch.nn.functional as F
from torch.utils import tensorboard as tb
from datetime import datetime

from models import DSRL
from models.losses import FALoss
from models.schedulers import PolynomialLR
from models.transforms import *
from metrices import *
from utils import *
import consts
import settings



def train_or_resume(is_resuming_training, device, distributed, mixed_precision, disable_cudnn_benchmark, num_workers, dataset, val_interval,
                    checkpoint_interval, checkpoint_history, init_weights, batch_size, epochs, learning_rate, end_learning_rate, momentum,
                    weights_decay, poly_power, stage, w1, w2, freeze_batch_norm, experiment_id, description, early_stopping, dry_run, **other_args):
    if distributed:
        # CAUTION: Setting manual seed for 'torch' library is important when executing distributed
        #          training as we need to make sure all weights of the model are initialized to the
        #          same value and hence the training is synchronous across processes.
        t.manual_seed(settings.RANDOM_SEED)

        t.distributed.init_process_group(distributed['BACKEND'],
                                         distributed['INIT_METHOD'],
                                         world_size=distributed['WORLD_SIZE'],
                                         rank=distributed['RANK'])
        if not t.distributed.is_initialized():
            raise RuntimeError("Couldn't initialize distributed process group!")

        is_master_rank = (distributed['RANK'] == 0)
        device_obj = t.device('cuda' if isCUDAdevice(device) else device, distributed['DEVICE_ID'] if isCUDAdevice(device) else 0)
    else:
        is_master_rank = True
        device_obj = t.device('cuda' if isCUDAdevice(device) else device)

    if is_master_rank:
        # Time keeper
        process_start_timestamp = datetime.now()

        if is_resuming_training:
            best_validation_dict = other_args['best_validation_dict']
        else:
            best_validation_dict = {'epoch': -1, 'best_miou_percent': 0., 'loss': 0.}

        # Prevent system from entering sleep state so that long training session is not interrupted
        if prevent_system_sleep():
            print(INFO("System will NOT be allowed to sleep until this training is complete/interrupted."))
        else:
            print(CAUTION("Please make sure system is NOT configured to sleep on idle! Sleep mode will pause training."))

    # Create model according to stage and optimizer
    model = DSRL(stage, dataset['settings']).to(device_obj)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=momentum,
                            weight_decay=weights_decay)

    if mixed_precision:
        # Enable mixed precision, if specified
        # CAUTION: It is recommended to call 'apex.amp.initialize()' before calling 'load_state_dict()' on model and optimizer.
        #          It should also precede 'torch.nn.parallel.DistributedDataParallel'.
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=mixed_precision)

    if is_resuming_training:
        model.load_state_dict(other_args['model_state_dict'], strict=True)
        optimizer.load_state_dict(other_args['optimizer_state_dict'])
        starting_epoch = other_args['epoch']
    else:
        starting_epoch = 0

        # Load initial weight, if any
        if init_weights:
            model.load_state_dict(load_checkpoint_or_weights(init_weights, map_location=device_obj)['model_state_dict'], strict=False)
        else:
            # Load checkpoint from previous stage, if not the first stage
            if stage == 1:
                if is_master_rank:
                    print(INFO("Pretrained weights for ResNet101 will be used to initialize network before training."))
                model.initialize_with_pretrained_weights(settings.WEIGHTS_ROOT_DIR)
            else:
                prev_weights_filename = os.path.join(experiment_id, settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE)
                if os.path.isfile(prev_weights_filename):
                    if is_master_rank:
                        print(INFO("'{:s}' weights file from previous stage was found and will be used to initialize network before training.".format(prev_weights_filename)))
                    weights_dict = load_checkpoint_or_weights(os.path.join(experiment_id, settings.WEIGHTS_DIR.format(stage=stage-1), settings.FINAL_WEIGHTS_FILE), map_location=device_obj)
                    model.load_state_dict(weights_dict['model_state_dict'], strict=False)
                else:
                    if is_master_rank:
                        print(CAUTION("'{:s}' weights file from previous stage was not found and network weights were initialized with Pytorch's default method.".format(prev_weights_filename)))

    if distributed:
        model = apex.parallel.DistributedDataParallel(model) if mixed_precision else t.nn.parallel.DistributedDataParallel(model, device_ids=[distributed['DEVICE_ID']])

    # Initialize training scheduler
    scheduler = PolynomialLR(optimizer,
                             max_decay_steps=epochs,
                             end_learning_rate=end_learning_rate,
                             power=poly_power,
                             last_epoch=(starting_epoch - 1))

    # Initialize loss functions and send them to device
    loss_funcs = [t.nn.CrossEntropyLoss(ignore_index=dataset['settings'].IGNORE_CLASS_LABEL),
                  t.nn.MSELoss(),
                  FALoss()]
    loss_funcs = [l.to(device_obj) for l in loss_funcs]

    # Prepare data from dataset
    os.makedirs(dataset['path'], exist_ok=True)
    if os.path.getsize(dataset['path']) == 0:
        raise Exception(FATAL("Cityscapes dataset was not found under '{:s}'.".format(dataset['path'])))

    # CAUTION: When 'num_workers' > 0, the compose classes in the following must be pick-able.
    #          So, no lambdas, numba etc in them.
    train_joint_transforms = JointCompose([JointRandomRotate(degrees=15.0, fill=(0, dataset['settings'].IGNORE_CLASS_LABEL)),
                                           JointRandomCrop(min_scale=1.0, max_scale=3.5),
                                           JointImageAndLabelTensor(dataset['settings'].LABEL_MAPPING_DICT),
                                           #JointColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                           JointHFlip(),
                                           # CAUTION: 'kernel_size' should be > 0 and odd integer
                                           JointRandomGaussianBlur(kernel_size=3, p=0.5),
                                           JointRandomGrayscale(p=0.1),
                                           JointNormalize(mean=dataset['settings'].MEAN, std=dataset['settings'].STD),
                                           JointScaledImage(new_img_size=DSRL.MODEL_INPUT_SIZE, new_seg_size=DSRL.MODEL_OUTPUT_SIZE)])
    train_dataset = dataset['class'](dataset['path'],
                                     split='train',
                                     transforms=train_joint_transforms)
    train_sampler = t.utils.data.DistributedSampler(train_dataset,
                                                    distributed['WORLD_SIZE'],
                                                    distributed['RANK'],
                                                    shuffle=True,
                                                    seed=settings.RANDOM_SEED,
                                                    drop_last=True) if distributed else None
    train_loader = t.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           num_workers=num_workers,
                                           pin_memory=isCUDAdevice(device),
                                           drop_last=True)

    if is_master_rank:
        val_joint_transforms = JointCompose([JointImageAndLabelTensor(dataset['settings'].LABEL_MAPPING_DICT),
                                             JointNormalize(mean=dataset['settings'].MEAN, std=dataset['settings'].STD),
                                             JointScaledImage(new_img_size=DSRL.MODEL_INPUT_SIZE, new_seg_size=DSRL.MODEL_OUTPUT_SIZE)])
        val_dataset = dataset['class'](dataset['path'],
                                       split='val',
                                       transforms=val_joint_transforms)
        val_loader = t.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=isCUDAdevice(device),
                                             drop_last=False)

    # Make sure proper log and weight directories exist
    train_logs_dir = os.path.join(experiment_id, settings.LOGS_DIR.format(stage=stage, mode='train'))
    val_logs_dir = os.path.join(experiment_id, settings.LOGS_DIR.format(stage=stage, mode='val'))
    os.makedirs(train_logs_dir, exist_ok=True)
    os.makedirs(val_logs_dir, exist_ok=True)

    # Start training and validation
    with ConditionalContextManager(is_master_rank, lambda: tb.SummaryWriter(log_dir=train_logs_dir)) as train_logger,\
         ConditionalContextManager(is_master_rank, lambda: tb.SummaryWriter(log_dir=val_logs_dir)) as val_logger:

        if is_master_rank:
            # Write training commandline parameters provided to 'params.txt' log file
            _write_params_file(os.path.join(train_logs_dir, settings.PARAMS_FILE),
                               "Timestamp: {:s}".format(process_start_timestamp.strftime("%c")),
                               "Device: {:s}".format(device),
                               "Distributed: {:}".format(distributed) if distributed else None,
                               "Mixed Precision: {:s}".format(mixed_precision) if mixed_precision else None,
                               "Disable CuDNN benchmark mode: {:}".format(disable_cudnn_benchmark) if isCUDAdevice(device) else None,
                               "No. of workers: {:d}".format(num_workers),
                               "Dataset: {:s}".format(dataset['name']),
                               "Dataset path: {:s}".format(dataset['path']),
                               "Validation interval: {:d}".format(val_interval),
                               "Checkpoint interval: {:d}".format(checkpoint_interval),
                               "Checkpoint history: {:d}".format(checkpoint_history),
                               "Initial weights: {:s}".format(init_weights) if init_weights else None,
                               "Resuming checkpoint: {:s}".format(other_args['checkpoint']) if is_resuming_training else None,
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
                               "Experiment ID: {:}".format(experiment_id) if experiment_id else None,
                               "Description: {:s}".format(description) if description else None,
                               "Early stopping: {:}".format(early_stopping))

            # Start training and then validation after specific intervals
            # Print number of training parameters
            print(INFO("Total training parameters: {:,}".format(countModelParams(model)[0])))
            train_logger.add_text("INFO", "Training started on {:s}.".format(process_start_timestamp.strftime("%c")), (starting_epoch + 1))
            log_string = "################################# Stage {:d} training STARTED #################################".format(stage)
            print('\n' + INFO(log_string))

            training_epoch_timetaken_list = []

        # Let's free as much unreferenced memory as possible before starting training
        gc.collect()
        if isCUDAdevice(device):
            t.cuda.empty_cache()
            t.cuda.synchronize(device_obj)

        for epoch in range((starting_epoch + 1), (epochs + 1)):
            if is_master_rank:
                log_string = "\n=> EPOCH {0:d}/{1:d}".format(epoch, epochs)
                print(log_string)

                training_epoch_begin_timestamp = datetime.now()

            # Do training for this epoch
            CE_train_avg_loss,\
            MSE_train_avg_loss,\
            FA_train_avg_loss,\
            Avg_train_loss,\
            _,\
            _ = _do_train_val(do_train=True,
                              epoch=epoch,
                              model=model,
                              dataset_settings=dataset['settings'],
                              device_obj=device_obj,
                              batch_size=batch_size,
                              stage=stage,
                              data_loader=train_loader,
                              loss_funcs=loss_funcs,
                              w1=w1,
                              w2=w2,
                              is_master_rank=is_master_rank,
                              logger=train_logger,
                              mixed_precision=mixed_precision,
                              freeze_batch_norm=freeze_batch_norm,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              dry_run=dry_run)

            if is_master_rank:
                # Auto save whole model between 'checkpoint_interval' epochs
                if checkpoint_history > 0 and epoch % checkpoint_interval == 0:
                    # Prepare some variables to save in checkpoint
                    model_state_dict = _get_state_dict(model)
                    optimizer_state_dict = optimizer.state_dict()
                    amp_state_dict = apex.amp.state_dict() if mixed_precision else None
                    CE_val_avg_loss = None
                    MSE_val_avg_loss = None
                    FA_val_avg_loss = None
                    Avg_val_loss = None

                    checkpoint_variables_dict = {}
                    for var in settings.VARIABLES_IN_CHECKPOINT:
                        checkpoint_variables_dict[var] = locals()[var]
                    save_checkpoint(os.path.join(experiment_id, settings.CHECKPOINTS_DIR.format(stage=stage)),
                                    settings.CHECKPOINT_FILE.format(epoch=epoch),
                                    **checkpoint_variables_dict)
                    print(INFO("Autosaved checkpoint for epoch {0:d} under '{1:s}'.".format(epoch,
                                                                                            settings.CHECKPOINTS_DIR.format(stage=stage))))

                    # Delete old autosaves, if any
                    checkpoint_epoch_to_delete = epoch - checkpoint_history * checkpoint_interval
                    if checkpoint_epoch_to_delete > 0:
                        checkpoint_to_delete_filename = os.path.join(experiment_id,
                                                                     settings.CHECKPOINTS_DIR.format(stage=stage),
                                                                     settings.CHECKPOINT_FILE.format(epoch=checkpoint_epoch_to_delete))
                        if os.path.isfile(checkpoint_to_delete_filename):
                            os.remove(checkpoint_to_delete_filename)

                # Do validation at epoch intervals of 'val_interval'
                if epoch % val_interval == 0:
                    CE_val_avg_loss,\
                    MSE_val_avg_loss,\
                    FA_val_avg_loss,\
                    Avg_val_loss,\
                    val_mIoU,\
                    val_accuracy = _do_train_val(do_train=False,
                                                 epoch=epoch,
                                                 model=model,
                                                 dataset_settings=dataset['settings'],
                                                 device_obj=device_obj,
                                                 batch_size=batch_size,
                                                 stage=stage,
                                                 data_loader=val_loader,
                                                 loss_funcs=loss_funcs,
                                                 w1=w1,
                                                 w2=w2,
                                                 is_master_rank=is_master_rank,
                                                 logger=val_logger,
                                                 mixed_precision=mixed_precision,
                                                 best_validation_dict=best_validation_dict,
                                                 dry_run=dry_run)

                    # Save epoch number, metrics and total error of best validation and then create a checkpoint
                    if val_mIoU > best_validation_dict['best_miou_percent']:
                        # Prepare some variables to save in checkpoint
                        best_validation_dict['epoch'] = epoch
                        best_validation_dict['best_miou_percent'] = val_mIoU
                        best_validation_dict['loss'] = Avg_val_loss
                        model_state_dict = _get_state_dict(model)
                        optimizer_state_dict = optimizer.state_dict()
                        amp_state_dict = apex.amp.state_dict() if mixed_precision else None

                        checkpoint_variables_dict = {}
                        for var in settings.VARIABLES_IN_CHECKPOINT:
                            checkpoint_variables_dict[var] = locals()[var]
                        save_checkpoint(os.path.join(experiment_id, settings.CHECKPOINTS_DIR.format(stage=stage)),
                                        settings.CHECKPOINT_FILE.format(epoch='_bestval'),
                                        **checkpoint_variables_dict)

                    # If early stopping is enabled, check if average training error is less than average
                    # validation error, and if so, stop training
                    if early_stopping and Avg_train_loss.avg < Avg_val_loss.avg:
                        log_string = "Early stopping was triggered at epoch {:d}.".format(epoch)
                        train_logger.add_text("INFO", log_string, epoch)
                        print(INFO(log_string))
                        break

            # Calculate new learning rate for next epoch
            scheduler.step()

            if is_master_rank:
                # Print estimated time for training completion
                training_epoch_timetaken_list.append((datetime.now() - training_epoch_begin_timestamp).total_seconds())
                training_epoch_avg_timetaken = np.mean(training_epoch_timetaken_list[(-val_interval*2):])   # NOTE: '*2' due to Nyquist sampling theorem
                print(INFO("Est. training completion in {:s}.".format(makeSecondsPretty(training_epoch_avg_timetaken * (epochs - epoch)))))

        if is_master_rank:
            # Save final training weights for this stage
            save_weights(os.path.join(experiment_id, settings.WEIGHTS_DIR.format(stage=stage)),
                         settings.FINAL_WEIGHTS_FILE,
                         _get_state_dict(model),
                         mixed_precision)

            process_end_timestamp = datetime.now()
            process_time_taken = (process_end_timestamp - process_start_timestamp).total_seconds()
            train_logger.add_text("INFO", "Training took {0:s} and completed on {1:s}.".format(makeSecondsPretty(process_time_taken),
                                                                                               process_end_timestamp.strftime("%c")),
                                  epochs)
            log_string = "################################# Stage {:d} training ENDED #################################".format(stage)
            print('\n' + INFO(log_string))


def _do_train_val(do_train, epoch, model, dataset_settings, device_obj, batch_size, stage, data_loader, loss_funcs, w1, w2, is_master_rank,
                  logger, mixed_precision, freeze_batch_norm=False, optimizer=None, scheduler=None, best_validation_dict=None, dry_run=False):
    # Set model to either training or testing mode
    model.train(mode=do_train)

    # If training and freeze BatchNorm layer option is ON, then freeze them
    if do_train and freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, settings.BATCHNORM_MODULE_CLASSES):
                module.eval()

    # Losses to report
    CE_avg_loss = AverageMeter('CE Avg. Loss')
    MSE_avg_loss = AverageMeter('MSE Avg. Loss')
    FA_avg_loss = AverageMeter('FA Avg. Loss')
    Avg_loss = AverageMeter('Avg. Loss')
    miou = mIoU(num_classes=dataset_settings.NUM_CLASSES)
    accuracy = Accuracy()

    with t.set_grad_enabled(mode=do_train),\
         ConditionalContextManager(is_master_rank, lambda: tqdm(total=len(data_loader),
                                                                desc='TRAINING' if do_train else 'VALIDATING',
                                                                colour='green' if do_train else 'yellow',
                                                                position=0 if do_train else 1,
                                                                leave=False,
                                                                bar_format=settings.PROGRESSBAR_FORMAT)) as progressbar:
        if not do_train and is_master_rank:
            # NOTE: We randomly select a batch index in validation to save input image and model's output
            #   to save in TensorBoard log.
            RANDOM_IMAGE_EXAMPLE_INDEX = np.random.randint(0, len(data_loader)//batch_size)

        for i, ((input_image, input_org), (target, target_org)) in enumerate(data_loader):
            # SANITY CHECK: Check data doesn't have any 'NaN' values
            assert not (t.isnan(input_image).any().item()),\
                FATAL("'input_scaled' contains 'NaN' values")
            assert not (False if input_org is None else t.isnan(input_org).any().item()),\
                FATAL("'input_org' contains 'NaN' values")
            assert not (t.isnan(target).any().item()),\
                FATAL("'target' contains 'NaN' values")

            input_image = input_image.to(device_obj)
            if stage > 1:
                input_org = input_org.to(device_obj)
            target = target.to(device_obj)
            if do_train:
                optimizer.zero_grad()

            SSSR_output, SISR_output, SSSR_transform_output, SISR_transform_output = model(input_image) if not dry_run else \
                [t.randn((target.shape[0], dataset_settings.NUM_CLASSES, *model.MODEL_OUTPUT_SIZE), device=device_obj, requires_grad=True),
                 t.randn((*input_org.shape[0:2], *model.MODEL_OUTPUT_SIZE), device=device_obj, requires_grad=True) if stage > 1 else t.zeros(1, requires_grad=False),
                 t.randn((input_image.shape[0], 1, 2, 2), device=device_obj, requires_grad=True) if stage > 2 else t.zeros(1, requires_grad=False),
                 t.randn((input_image.shape[0], 1, 2, 2), device=device_obj, requires_grad=True) if stage > 2 else t.zeros(1, requires_grad=False)]
            # SANITY CHECK: Check network outputs doesn't have any 'NaN' values
            assert not t.isnan(SSSR_output).any().item(),\
                FATAL("SSSR network output contains 'NaN' values and so cannot continue.")
            assert not t.isnan(SISR_output).any().item(),\
                FATAL("SISR network output contains 'NaN' values and so cannot continue.")
            assert not t.isnan(SSSR_transform_output).any().item(),\
                FATAL("SSSR feature transform network output contains 'NaN' values and so cannot continue.")
            assert not t.isnan(SISR_transform_output).any().item(),\
                FATAL("SISR feature transform network output contains 'NaN' values and so cannot continue.")

            # Resize output of SSSR layer to the same size as 'target', if required
            #if SSSR_output.shape[-2:] != target.shape[-2:]:
            #    SSSR_output = F.interpolate(SSSR_output, size=target.shape[-2:], mode='nearest')

            # Resize output of SISR layer to the same size as 'input_org', if required
            if stage > 1 and SISR_output.shape[-2:] != input_org[-2:]:
                SISR_output = F.interpolate(SISR_output, size=input_org.shape[-2:], mode='bilinear', align_corners=True)

            CE_loss = loss_funcs[0](SSSR_output, target.long())
            MSE_loss = (w1 * loss_funcs[1](SISR_output, input_org)) if stage > 1 else t.tensor(0., requires_grad=False)
            FA_loss = (w2 * loss_funcs[2](SSSR_transform_output, SISR_transform_output)) if stage > 2 else t.tensor(0., requires_grad=False)
            Total_loss = CE_loss + MSE_loss + FA_loss

            if do_train and not dry_run:
                with ConditionalContextManager(mixed_precision,
                                               lambda: apex.amp.scale_loss(Total_loss, optimizer, model=model),
                                               lambda: Total_loss) as scaled_total_loss:
                    scaled_total_loss.backward()     # Backpropagate
                optimizer.step()          # Increment global step

            # Convert loss tensors to float on CPU memory
            CE_loss = CE_loss.item()
            MSE_loss = MSE_loss.item()
            FA_loss = FA_loss.item()
            Total_loss = Total_loss.item()

            # Compute averages for losses
            # CAUTION: During training, 'drop_last' argument of 'DataLoader' will be True, so
            #          the number of batched input might NOT be equal to 'batch_size'. Hence, to
            #          avoid small error while averaging errors we use 'input_scaled.shape[0]' instead.
            CE_avg_loss.update(CE_loss, input_image.shape[0])
            MSE_avg_loss.update(MSE_loss, input_image.shape[0])
            FA_avg_loss.update(FA_loss, input_image.shape[0])
            Avg_loss.update(Total_loss, input_image.shape[0])

            if is_master_rank:
                # Add loss information to progress bar
                log_string = []
                log_string.append("CE: {:.4f}".format(CE_avg_loss.avg))
                if stage > 1:
                    log_string.append("MSE: {:.4f}".format(MSE_avg_loss.avg))
                    if stage > 2:
                        log_string.append("FA: {:.4f}".format(FA_avg_loss.avg))
                    log_string.append("Total: {:.3f}".format(Avg_loss.avg))
                log_string = ', '.join(log_string)
                progressbar.set_postfix_str("[{:s}]".format(log_string))
                progressbar.update()

                # Collect metrics for validation mode
                if not do_train:
                    pred = t.argmax(SSSR_output, dim=1).detach().cpu().numpy()
                    target = target.detach().cpu().numpy()
                    valid_labels_mask = (target != dataset_settings.IGNORE_CLASS_LABEL)    # Boolean mask
                    accuracy.update(pred, target, valid_labels_mask)
                    miou.update(pred, target, valid_labels_mask)

                    # On validation mode, if current data index matches 'RANDOM_IMAGE_EXAMPLE_INDEX', save visualization to TensorBoard
                    if i == RANDOM_IMAGE_EXAMPLE_INDEX:
                        if input_org.shape[-2:] != target.shape[-2:]:
                            input_org = F.interpolate(input_org, size=target.shape[-2:], mode='bilinear', align_corners=True)
                        input_org = input_org.detach().cpu().numpy()[0]
                        input_org = np.array(dataset_settings.STD).reshape(consts.NUM_RGB_CHANNELS, 1, 1) * input_org +\
                                    np.array(dataset_settings.MEAN).reshape(consts.NUM_RGB_CHANNELS, 1, 1)
                        input_org = np.clip(input_org * 255., a_min=0.0, a_max=255.).astype(np.uint8)
                        SSSR_output = np.argmax(SSSR_output.detach().cpu().numpy()[0], axis=0)    # Bring back result to CPU memory and select first in batch
                        logger.add_image("EXAMPLE",
                                         make_input_output_visualization(input_org, SSSR_output, dataset_settings.CLASS_RGB_COLOR),
                                         epoch)

        if is_master_rank:
            # Log training losses for this epoch to TensorBoard
            logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_avg_loss.avg, epoch)
            if stage > 1:
                logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_avg_loss.avg, epoch)
                if stage > 2:
                    logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_avg_loss.avg, epoch)
                logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_loss.avg, epoch)

            if do_train:
                # Log learning rate for this epoch to TensorBoard
                logger.add_scalar("Stage {:d}/Learning rate".format(stage), scheduler.get_last_lr()[0], epoch)
            else:
                logger.add_scalar("Stage {:d}/Accuracy %".format(stage), accuracy(), epoch)
                logger.add_scalar("Stage {:d}/mIoU %".format(stage), miou(), epoch)

            # Show learning rate and average losses before ending epoch
            log_string = []
            log_string.append("Avg. CE: {:.4f}".format(CE_avg_loss.avg))
            if stage > 1:
                log_string.append("Avg. MSE: {:.4f}".format(MSE_avg_loss.avg))
                if stage > 2:
                    log_string.append("Avg. FA: {:.4f}".format(FA_avg_loss.avg))
                log_string.append("Total Avg. Loss: {:.3f}".format(Avg_loss.avg))

            if do_train:
                log_string.append("Learning Rate: {:6f}".format(scheduler.get_last_lr()[0]))
            else:
                log_string.append("Accuracy %: {:.2f}".format(accuracy()))
                log_string.append("mIoU %: {:.2f}".format(miou()))
                log_string.append("Best mIoU % so far is {:.2f} at epoch {:d}.".format(max(miou(), best_validation_dict['best_miou_percent']),
                                                                                       epoch if miou() > best_validation_dict['best_miou_percent'] else best_validation_dict['epoch']))

            log_string = ', '.join(log_string)
            if do_train:
                print(log_string)
            else:
                print(termcolor.colored("Validation results:\n{:s}".format(log_string), 'yellow'))

    return CE_avg_loss.avg, MSE_avg_loss.avg, FA_avg_loss.avg, Avg_loss.avg, miou(), accuracy()


def _write_params_file(filename, *list_params):
    list_params = list(filter(lambda x: x is not None, list_params))    # Remove all 'None' items
    with open(filename, mode='w') as params_file:
        params_file.write('\n'.join(list_params))   # NOTE: '\n' here automatically converts it to newline for the current platform

def _get_state_dict(model):
    # NOTE: This method is required because we need 'state_dict' of our model and NOT 'DistributedDataParallel'
    return model.module.state_dict() if isinstance(model, (t.nn.parallel.DistributedDataParallel, apex.parallel.DistributedDataParallel)) else model.state_dict()