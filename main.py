import sys
import os
import os.path
import argparse
from tqdm.auto import tqdm as tqdm
import torch as t

from models import DSRLSS
from models.losses import FALoss
from utils import *
import settings


def do_train_val(do_train: bool, model, device, stage, data_loader, w1=None, w2=None, optimizer=None):
    model.train(mode=do_train)

    with t.set_grad_enabled(mode=do_train):
        with tqdm(total=len(data_loader),
                  desc='TRAINING' if do_train else 'VALIDATION',
                  colour='green' if do_train else 'yellow',
                  position=0 if do_train else 1,
                  leave=False, bar_format=settings.PROGRESSBAR_FORMAT) as progressbar:
            CE_avg_loss = AverageMeter('CE Avg. Loss')
            MSE_avg_loss = AverageMeter('MSE Avg. Loss')
            FA_avg_loss = AverageMeter('FA Avg. Loss')
            Avg_loss = AverageMeter('Avg. Loss')

            for (input_, target) in data_loader:
                input_, target = input_.to(device), target.to(device)
                if do_train:
                    optimizer.zero_grad()

                SSSR_output, SISR_output = model.forward(input_)
                CE_loss = t.nn.CrossEntropyLoss()(SSSR_output, target)
                if stage > 1:
                    MSE_loss = (w1 * t.nn.MSELoss()(SISR_output, target)) if stage > 1 else t.tensor(0., requires_grad=False)
                    if stage == 3:
                        FA_loss = (w2 * FALoss()(SSSR_output, SISR_output)) if stage > 2 else t.tensor(0., requires_grad=False)
                    loss = CE_loss + \
                            (MSE_loss if stage > 1 else t.torch(0., requires_grad=False)) + \
                            (FA_loss if stage == 2 else t.torch(0., requires_grad=False))

                if do_train:
                    loss.backward()     # Backpropagate
                    optimizer.step()    # Increment global step

                # Convert losses to float on CPU memory
                CE_loss = CE_loss.item()
                if stage > 1:
                    MSE_loss = MSE_loss.item()
                    if stage == 3:
                        FA_loss = FA_loss.item()
                    loss = loss.item()

                # Compute averages for losses
                CE_avg_loss.update(CE_Loss, input_.size(0))
                if stage > 1:
                    MSE_avg_loss.update(MSE_loss, input_.size(0))
                    if stage == 2:
                        FA_avg_loss.update(FA_loss, input_.size(0))
                    Avg_loss.update(loss, input_.size(0))

                # Add loss information to progress bar
                log_string = []
                log_string.append("CE: {:.3f}".format(CE_loss))
                if stage > 1:
                    log_string.append("MSE: {:.3f}".format(MSE_loss))
                    if stage == 3:
                        log_string.append("FA: {:.3f}".format(FA_loss))
                    log_string.append("Total: {:.3f}".format(loss))
                log_string = ', '.join(log_string)
                progressbar.set_postfix_str("Losses [{0}]".format(log_string))
                progressbar.update()

            # Show average losses before ending epoch
            log_string = []
            log_string.append("CE: {:.3f}".format(CE_avg_loss.avg))
            if stage > 1:
                log_string.append("MSE: {:.3f}".format(MSE_avg_loss.avg))
                if stage == 3:
                    log_string.append("FA: {:.3f}".format(FA_avg_loss.avg))
                log_string.append("Total: {:.3f}".format(Avg_loss.avg))
            log_string = ', '.join(log_string)
            tqdm.write(log_string)

            return CE_avg_loss, MSE_avg_loss, FA_avg_loss, Avg_loss


def main(train,
         test_file=None,
         device=None,
         num_workers=None,
         val_interval=None,
         batch_size=None,
         epochs=None,
         learning_rate=None,
         momentum=None,
         weight_decay=None,
         poly_power=None,
         stage=None,
         w1=None,
         w2=None,
         description=None):
    # Create model according to stage
    model = DSRLSS(stage)

    if train:
        # Training and Validation on dataset mode
        
        # Module imports
        import torchvision as tv
        from torch.utils import tensorboard as tb

        # Prepare data from CityScapes dataset
        os.makedirs(settings.CITYSCAPES_DATASET_DATA_DIR, exist_ok=True)
        if os.path.getsize(settings.CITYSCAPES_DATASET_DATA_DIR) == 0:
            tqdm.write("Cityscapes dataset was not found under '{0:s}'.".format(settings.CITYSCAPES_DATASET_DATA_DIR))
            return

        #input_transforms = tv.transforms.Compose([tv.transforms.Normalize(mean=settings.CITYSCAPES_DATASET_MEAN, std=settings.CITYSCAPES_DATASET_STD),
        #                                          tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)])
        #joint_transforms = tv.transforms.Compose([tv.transforms.PILToTensor(),
        #                                          tv.transforms.RandomHorizontalFlip(),
        #                                          tv.transforms.RandomResizedCrop(size=DSRLSS.MODEL_INPUT_SIZE)])
        train_input_transforms = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                        tv.transforms.Normalize(mean=settings.CITYSCAPES_DATASET_MEAN, std=settings.CITYSCAPES_DATASET_STD),
                                                        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)])
        val_input_transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=settings.CITYSCAPES_DATASET_MEAN, std=settings.CITYSCAPES_DATASET_STD)])
        target_transforms = tv.transforms.Compose([tv.transforms.ToTensor()])
        train_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                               split='train',
                                               mode='fine',
                                               target_type='semantic',
                                               transform=train_input_transforms,
                                               target_transform=target_transforms,
                                               transforms=None)
        train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                             split='val',
                                             mode='fine',
                                             target_type='semantic',
                                             transform=val_input_transform,
                                             target_transform=target_transforms,
                                             transforms=None)
        val_loader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Make sure proper log directories exist
        train_logs_dir = settings.LOGS_DIR.format(stage=stage, mode='train')
        val_logs_dir = settings.LOGS_DIR.format(stage=stage, mode='val')
        os.makedirs(train_logs_dir, exist_ok=True)
        os.makedirs(val_logs_dir, exist_ok=True)

        # Write training parameters provided to params.txt log file
        with open(os.path.join(train_logs_dir, 'params.txt'), mode='w') as params_file:
            params_file.writelines("Device: {:s}".format(str(device)))
            params_file.writelines("Validation interval: {:d}".format(val_interval))
            params_file.writelines("Batch size: {:d}".format(batch_size))
            params_file.writelines("Epochs: {:d}".format(epochs))
            params_file.writelines("Learning rate: {:f}".format(learning_rate))
            params_file.writelines("Momentum: {:f}".format(momentum))
            params_file.writelines("Weight decay: {:f}".format(weight_decay))
            params_file.writelines("Poly power: {:f}".format(poly_power))
            params_file.writelines("Stage: {:d}".format(stage))
            if stage > 1:
                params_file.writelines("Loss Weight 1: {:f}".format(w1))

                if stage == 2:
                    params_file.writelines("Loss Weight 2: {:f}".format(w2))
            params_file.writelines("Description: {:s}".format(description))

        with tb.SummaryWriter(log_dir=train_logs_dir) as train_logger, \
             tb.SummaryWriter(log_dir=val_logs_dir) as val_logger:

            # Training optimizer and schedular
            optimizer = t.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
            scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=poly_power)

            # Load weights from previous stages, if any
            if stage > 1:
                model.load_state_dict(t.load(os.path.join(settings.WEIGHTS_DIR.format(stage=1), settings.WEIGHTS_FILE)))
                if stage == 3:
                    model.load_state_dict(t.load(os.path.join(settings.WEIGHTS_DIR.format(stage=2), settings.WEIGHTS_FILE)))

            # Copy the model into 'device'
            model = model.to(device)

            # Start training and then validation after specific intervals
            log_string = "\n################################# Stage {:d} training STARTED #################################".format(stage)
            tqdm.write(log_string)

            for epoch in range(1, epochs + 1):
                log_string = "\nEPOCH {0:d}/{1:d}".format(epoch, epochs)
                tqdm.write(log_string)

                # Do training for this epoch
                CE_train_avg_loss, \
                MSE_train_avg_loss, \
                FA_train_avg_loss, \
                Avg_train_loss = do_train_val(do_train=True,
                                              model=model,
                                              device=device,
                                              stage=stage,
                                              data_loader=train_loader,
                                              w1=w1,
                                              w2=w2,
                                              optimizer=optimizer)

                # Log training losses for this epoch to TensorBoard
                train_logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_train_avg_loss.avg, epoch)
                if stage > 1:
                    train_logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_train_avg_loss.avg, epoch)
                    if stage == 3:
                        train_logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_train_avg_loss.avg, epoch)
                    train_logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_train_loss.avg, epoch)

                # Log learning rate for this epoch to TensorBoard
                train_logger.add_scalar("Stage {:d}/Learning rate".format(stage), scheduler.get_last_lr(), epoch)

                if (epoch + 1) % val_interval == 0:
                    # Do validation at epoch intervals of 'val_interval'
                    CE_val_avg_loss, \
                    MSE_val_avg_loss, \
                    FA_val_avg_loss, \
                    Avg_val_loss = do_train_test(do_train=False,
                                                 model=model,
                                                 device=device,
                                                 stage=stage,
                                                 data_loader=val_loader)

                    # Log validation losses for this epoch to TensorBoard
                    val_logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_val_avg_loss.avg, epoch)
                    if stage > 1:
                        val_logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_val_avg_loss.avg, epoch)
                        if stage == 3:
                            val_logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_val_avg_loss.avg, epoch)
                        val_logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_val_loss.avg, epoch)

            # Save training weights for this stage
            os.makedirs(settings.WEIGHTS_DIR.format(stage=stage))
            t.save(model.state_dict(), os.path.join(settings.WEIGHTS_DIR.format(stage=stage), settings.WEIGHTS_FILE))

            log_string = "\n################################# Stage {:d} training ENDED #################################".format(stage)
            tqdm.write(log_string)
    else:
        # Evaluation/Testing on input image mode

        # Module imports
        import numpy as np
        from PIL import Image, ImageOps

        model.eval()

        # Load weights for specified stage
        tqdm.write("INFO: Using weights from stage{:d} training.".format(stage))
        model.load_state_dict(t.load(os.path.join(settings.WEIGHTS_DIR.format(stage=stage), settings.WEIGHTS_FILE)))

        # Copy the model into 'device'
        model = model.to(device)

        # Load image file, rotate according to EXIF info, add 'batch' dimension and convert to tensor
        input_image = ImageOps.exif_transpose(Image.open(test_file)).resize(settings.INPUT_SIZE).convert('RGB')

        with t.no_grad():
            input_image_tensor = t.tensor(np.transpose(np.expand_dims(np.array(input_image, dtype=np.float32), axis=0), (0, 3, 1, 2)),
                                            device=device, requires_grad=False)
            SSSR_output, _ = model.forward(input_image_tensor)  # Add batch dimension and change to (B, C, H, W)
            SSSR_output = np.squeeze(SSSR_output.detach().cpu().numpy(), axis=0)    # Bring back result to CPU memory and remove batch dimension

        # Prepare output image consisting of model input and segmentation image side-by-side
        output_image = np.zeros((settings.INPUT_SIZE[0] * 2, settings.INPUT_SIZE[1], 3), dtype=np.uint8)

        for x in range(settings.INPUT_SIZE[1]):
            for y in range(settings.INPUT_SIZE[0]):
                output_image[x, y, :] = input_image.getpixel((x, y))
                output_image[x + settings.INPUT_SIZE[1], y + settings.INPUT_SIZE[0], :] = getRGBColorFromClass(np.argmax(SSSR_output, axis=0))
        output_image = Image.fromarray(output_image, mode='RGB')    # Convert from numpy array to PIL Image

        # Save and show output on plot
        os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
        output_image_filename = os.path.join(settings.OUTPUTS_DIR, os.path.splitext(os.path.basename(test_file))[0] + '.png')

        output_image.save(output_image_filename, format='PNG')
        tqdm.write("Output image saved in: {0:s}".format(output_image_filename))

        output_image.show(title='Segmentation output')
        


if __name__ == '__main__':
    assert check_version(sys.version_info, *settings.MIN_PYTHON_VERSION), \
        "This program needs at least Python {0:d}.{1:d} interpreter.".format(*settings.MIN_PYTHON_VERSION)
    assert check_version(t.__version__, *settings.MIN_PYTORCH_VERSION), \
        "This program needs at least PyTorch {0:d}.{1:d}.".format(*settings.MIN_PYTORCH_VERSION)

    parser = argparse.ArgumentParser(description="Implementation of 'Dual Super Resolution Learning For Segmantic Segmentation' CVPR 2020 paper.")
    parser.add_argument('--train', action='store_true', default=False, help="Train the model")
    parser.add_argument('--test_file', type=str, help="Run evaluation on a image file using trained weights")
    parser.add_argument('--device', default='gpu', type=str.lower, choices=['cpu', 'gpu'], help="Device to create model in")
    parser.add_argument('--num_workers', default=4, type=int, help="Number of workers for data loader")
    parser.add_argument('--val_interval', default=10, type=int, help="Epoch intervals after which to perform validation")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size to use for training and testing")
    parser.add_argument('--epochs', type=int, help="Number of epochs to train")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum value for SGD")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for SGD")
    parser.add_argument('--poly_power', type=float, default=0.9, help="Power for poly learning rate strategy")
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True, help="0: Train SSSR only\n1: Train SSSR+SISR\n2: Train SSSR+SISR with feature affinity")
    parser.add_argument('--w1', type=float, default=0.1, help="Weight for MSE loss")
    parser.add_argument('--w2', type=float, default=1.0, help="Weight for FA loss")
    parser.add_argument('--description', type=str, default='', help="Description text to save with params.txt")
    args = parser.parse_args()

    # Validate arguments according to mode
    if args.train:
        if not args.num_workers >= 0:
            raise argparse.ArgumentTypeError("'num_workers' should be greater than or equal to 0!")

        if not args.val_interval > 0:
            raise argparse.ArgumentTypeError("'val_interval' should be greater than 0!")

        if not args.batch_size > 0:
            raise argparse.ArgumentTypeError("'batch_size' should be greater than 0!")

        if not args.epochs > 0:
            raise argparse.ArgumentTypeError("'epochs' should be greater than 0!")

        if not args.learning_rate > 0.:
            raise argparse.ArgumentTypeError("'learning_rate' should be greater than 0!")

        if not args.momentum > 0.:
            raise argparse.ArgumentTypeError("'momentum' should be greater than 0!")

        if not args.weight_decay > 0.:
            raise argparse.ArgumentTypeError("'weight_decay' should be greater than 0!")

        if not args.poly_power > 0.:
            raise argparse.ArgumentTypeError("'poly_power' should be greater than 0!")
    else:
        if args.test_file is not None:
            raise argparse.ArgumentTypeError("'test_file' is required when 'train' parameter is false!")

        if not os.path.isfile(args.test_file):
            raise argparse.ArgumentTypeError("File specified in 'test_file' parameter doesn't exists!")

        if not os.path.isfile(settings.WEIGHTS_DIR.format(stage=stage), settings.WEIGHTS_FILE):
            raise argparse.ArgumentTypeError("Weight file '{0:s}' for stage{1:d} was not found!"\
                    .format(os.path.join(settings.WEIGHTS_DIR.format(stage=stage), settings.WEIGHTS_FILE), args.stage))

    main(train=args.train,
         test_file=args.test_file,
         device=t.device('cuda' if args.device == 'gpu' else args.device),
         num_workers=args.num_workers,
         val_interval=args.val_interval,
         batch_size=args.batch_size,
         epochs=args.epochs,
         learning_rate=args.learning_rate,
         momentum=args.momentum,
         weight_decay=args.weight_decay,
         poly_power=args.poly_power,
         stage=args.stage,
         w1=args.w1,
         w2=args.w2,
         description=args.description)