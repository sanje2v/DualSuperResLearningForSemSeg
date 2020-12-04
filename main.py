import sys
import os
import os.path
import argparse
import torch as t

from models import DSRLSS
from utils import *
import settings



def do_train(model, device, log_interval, stage, w1, w2,
             train_loader, optimizer, train_logger, PROGRESSBAR_FORMAT):
    model.train()

    with tqdm(train_loader, desc='TRAINING', colour='green', position=0,
              leave=False, bar_format=PROGRESSBAR_FORMAT) as progressbar:
        CE_avg_loss = AverageMeter('CE Avg. Loss')
        MSE_avg_loss = AverageMeter('MSE Avg. Loss')
        FA_avg_loss = AverageMeter('FA Avg. Loss')
        Avg_loss = AverageMeter('Avg. Loss')

        for train_batch_idx, (train_input, train_target) in enumerate(progressbar):
            train_input, train_target = getattr(train_input, device)(), getattr(train_target, device)()
            optimizer.zero_grad()

            SSSR_output, SISR_output = model.forward(train_input)
            CE_loss = t.nn.CrossEntropyLoss()(SSSR_output, train_target)
            MSE_loss = (w1 * t.nn.MSELoss()(SISR_output, train_target)) if stage > 1 else t.tensor(0., requires_grad=False)
            FA_loss = (w2 * FELoss()(SSSR_output, SISR_output)) if stage > 2 else t.tensor(0., requires_grad=False)
            loss = CE_loss + MSE_loss + FA_loss

            loss.backward()     # Backpropagate
            optimizer.step()    # Increment global step

            # Convert losses to float on CPU memory
            CE_loss = CE_loss.item()
            MSE_loss = MSE_loss.item()
            FA_loss = FA_loss.item()
            loss = loss.item()

            # Compute averages for losses
            CE_avg_loss.update(CE_Loss, train_input.size(0))
            MSE_avg_loss.update(MSE_loss, train_input.size(0))
            FA_avg_loss.update(FA_loss, train_input.size(0))
            Avg_loss.update(loss, train_input.size(0))

            # Add loss information to progress bar
            log_string = []
            if stage in [1, 3]:
                log_string.append("CE: {:.3f}".format(CE_loss))
            if stage > 1:
                log_string.append("MSE: {:.3f}".format(MSE_loss))
            if stage == 2:
                log_string.append("FA: {:.3f}".format(FA_loss))
            if stage > 1:
                log_string.append("Total: {:3f}".format(loss))
            log_string = ', '.join(log_string)
            progressbar.set_postfix_str("Losses [{0}]".format(log_string))
            progressbar.update()

            if (train_batch_idx + 1) % log_interval == 0:
                pass

        # Show average losses before ending epoch
        log_string = []
        if stage in [1, 3]:
            log_string.append("CE: {:.3f}".format(CE_avg_loss.avg))
        if stage > 1:
            log_string.append("MSE: {:.3f}".format(MSE_avg_loss.avg))
        if stage == 2:
            log_string.append("FA: {:.3f}".format(FA_avg_loss.avg))
        if stage > 1:
            log_string.append("Total: {:3f}".format(Avg_loss.avg))
        log_string = ', '.join(log_string)
        tqdm.write(log_string)

        return CE_avg_loss, MSE_avg_loss, FA_avg_loss, Avg_loss


def do_val(model, device, log_interval, stage, w1, w2,
           val_loader, val_logger, PROGRESSBAR_FORMAT):
    model.eval()

    with t.no_grad():
        with tqdm(val_loader, desc='VALIDATION', colour='yellow', position=1,
                  leave=False, bar_format=PROGRESSBAR_FORMAT) as progressbar:
            CE_avg_loss = AverageMeter('CE Avg. Loss')
            MSE_avg_loss = AverageMeter('MSE Avg. Loss')
            FA_avg_loss = AverageMeter('FA Avg. Loss')
            Avg_loss = AverageMeter('Avg. Loss')
            
            for val_batch_idx, (val_input, val_target) in enumerate():
                val_input, val_target = getattr(val_input, device)(), getattr(val_target, device)()

                SSSR_output, SISR_output = model.forward(val_input)
                CE_loss = t.nn.CrossEntropyLoss()(SSSR_output, val_target)
                MSE_loss = (w1 * t.nn.MSELoss()(SISR_output, val_target)) if stage > 1 else t.tensor(0., requires_grad=False)
                FA_loss = (w2 * FELoss()(SSSR_output, SISR_output)) if stage > 2 else t.tensor(0., requires_grad=False)
                loss = CE_loss + MSE_loss + FA_loss

                # Convert losses to float on CPU memory
                CE_loss = CE_loss.item()
                MSE_loss = MSE_loss.item()
                FA_loss = FA_loss.item()
                loss = loss.item()

                # Compute averages for losses
                CE_avg_loss.update(CE_Loss, val_input.size(0))
                MSE_avg_loss.update(MSE_loss, val_input.size(0))
                FA_avg_loss.update(FA_loss, val_input.size(0))
                Avg_loss.update(loss, val_input.size(0))

                # Add loss information to progress bar
                log_string = []
                if stage in [1, 3]:
                    log_string.append("CE: {:.3f}".format(CE_loss))
                if stage > 1:
                    log_string.append("MSE: {:.3f}".format(MSE_loss))
                if stage == 2:
                    log_string.append("FA: {:.3f}".format(FA_loss))
                if stage > 1:
                    log_string.append("Total: {:3f}".format(loss))
                log_string = ', '.join(log_string)
                progressbar.set_postfix_str("Losses [{0}]".format(log_string))
                progressbar.update()

                if (val_batch_idx + 1) % log_interval == 0:
                    pass

            # Show average losses before ending epoch
            log_string = []
            if stage in [1, 3]:
                log_string.append("CE: {:.3f}".format(CE_avg_loss.avg))
            if stage > 1:
                log_string.append("MSE: {:.3f}".format(MSE_avg_loss.avg))
            if stage == 2:
                log_string.append("FA: {:.3f}".format(FA_avg_loss.avg))
            if stage > 1:
                log_string.append("Total: {:3f}".format(Avg_loss.avg))
            log_string = ', '.join(log_string)
            tqdm.write(log_string)

            return CE_avg_loss, MSE_avg_loss, FA_avg_loss, Avg_loss


def main(train,
         eval_file=None,
         device='cpu',
         log_interval=None,
         val_interval=None,
         batch_size=None,
         epochs=None,
         learning_rate=None,
         momentum=None,
         weight_decay=None,
         poly_power=None,
         stage=None,
         w1=None,
         w2=None):
    # Create model according to stage
    model = DSRLSS(train, stage)

    if train:
        # Training and Validation on dataset mode
        
        # Module imports
        import logging
        from tqdm.auto import tqdm
        from torch.utils import tensorboard as tb

        from losses import FELoss

        # Settings
        PROGRESSBAR_FORMAT = '{desc}: {percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}'
        WEIGHTS_DIR = 'weights'
        LOGS_DIR = 'logs'

        # Prepare data
        train_loader = None
        val_loader = None
        
        # Make sure proper directories exist
        os.makedirs(os.path.join(WEIGHTS_DIR, "stage{:d}".format(stage)))
        os.makedirs(LOGS_DIR, exist_ok=True)

        with tb.SummaryWriter(log_dir=os.path.join(LOGS_DIR, "stage{:d}".format(stage), "train")) as train_logger, \
             tb.SummaryWriter(log_dir=os.path.join(LOGS_DIR, "stage{:d}".format(stage), "val")) as val_logger:

            # Training optimizer and schedular
            optimizer = t.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
            scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=poly_power)

            # Load weights from previous stages, if any
            if stage == 3:
                model.load_state_dict(t.load(os.path.join(WEIGHTS_DIR, "stage1")))
                model.load_state_dict(t.load(os.path.join(WEIGHTS_DIR, "stage2")))

            # Copy the model into 'device'
            model = getattr(model, device)()

            # Start training and then validation in between intervals
            log_string = "\n################################# Starting stage {:d} training #################################".format(stage)
            tqdm.write(log_string)

            for epoch in range(1, epochs + 1):
                log_string = "\nEPOCH {0:d}/{1:d}".format(epoch, epochs)
                tqdm.write(log_string)

                # Do training for this epoch
                CE_train_avg_loss, \
                MSE_train_avg_loss, \
                FA_train_avg_loss, \
                Avg_train_loss = do_train(model, device, log_interval, stage, w1, w2, train_loader, optimizer, train_logger, PROGRESSBAR_FORMAT)

                # Log training losses for this epoch to TensorBoard
                if stage in [1, 3]:
                    train_logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_train_avg_loss.avg, epoch)
                if stage > 1:
                    train_logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_train_avg_loss.avg, epoch)
                if stage == 2:
                    train_logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_train_avg_loss.avg, epoch)
                if stage > 1:
                    train_logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_train_loss.avg, epoch)

                if (epoch + 1) % val_interval == 0:
                    # Do validation at epoch intervals of 'val_interval'
                    CE_val_avg_loss, \
                    MSE_val_avg_loss, \
                    FA_val_avg_loss, \
                    Avg_val_loss = do_val(model, device, log_interval, stage, w1, w2, val_loader, optimizer, val_logger, PROGRESSBAR_FORMAT)

                    # Log training losses for this epoch to TensorBoard
                    if stage in [1, 3]:
                        val_logger.add_scalar("Stage {:d}/CE Loss".format(stage), CE_val_avg_loss.avg, epoch)
                    if stage > 1:
                        val_logger.add_scalar("Stage {:d}/MSE Loss".format(stage), MSE_val_avg_loss.avg, epoch)
                    if stage == 2:
                        val_logger.add_scalar("Stage {:d}/FA Loss".format(stage), FA_val_avg_loss.avg, epoch)
                    if stage > 1:
                        val_logger.add_scalar("Stage {:d}/Total Loss".format(stage), Avg_val_loss.avg, epoch)

            log_string = "\n################################# Ending stage {:d} training #################################".format(stage)
            tqdm.write(log_string)
    else:
        # Evaluation/Testing on input image mode

        # Module imports
        import numpy as np
        import imageio

        model.eval()

        # Load image file, add 'batch' dimension and convert to tensor
        input_ = t.from_numpy(np.expand_dims(imageio.imread(eval_file), axis=0))
        with t.no_grad():
            SSSR_output, _ = model.forward(input_)


if __name__ == '__main__':
    assert check_version(sys.version_info, major=3, minor=7), "This program needs at least Python 3.7 interpreter."
    assert check_version(t.__version__, major=1, minor=7), "This program needs at least PyTorch 1.7."

    parser = argparse.ArgumentParser(description="Implementation of 'Dual Super Resolution Learning For Segmantic Segmentation' CVPR 2020 paper.")
    parser.add_argument('--train', action='store_true', default=False, help="Train the model")
    parser.add_argument('--eval_file', type=str, help="Run evaluation on a image file using trained weights")
    parser.add_argument('--device', default='gpu', type=str.lower, choices=['cpu', 'gpu'], help="Device to create model in")
    parser.add_argument('--log_interval', default=10, type=int, help="Batch intervals in a training after which to log to file")
    parser.add_argument('--val_interval', default=10, type=int, help="Epoch intervals after which to perform validation")
    parser.add_argument('--batch_size', default=8, type=int, help="Batch size to use for training and testing")
    parser.add_argument('--epochs', type=int, help="Number of epochs to train")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum value for SGD")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay for SGD")
    parser.add_argument('--poly_power', type=float, default=0.9, help="Power for poly learning rate strategy")
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3], help="0: Train SSSR only\n1: Train SISR only\n2: Train SSSR and SISR jointly")
    parser.add_argument('--w1', type=float, default=0.1, help="Weight for MSE loss")
    parser.add_argument('--w2', type=float, default=1.0, help="Weight for FA loss")
    args = parser.parse_args()

    # Validate arguments according to mode
    if args.train:
        if args.log_interval is None or not args.log_interval > 0:
            raise argparse.ArgumentTypeError("'log_interval' should be greater than 0!")

        if args.val_interval is None or not args.val_interval > 0:
            raise argparse.ArgumentTypeError("'val_interval' should be greater than 0!")

        if args.batch_size is None or not args.batch_size > 0:
            raise argparse.ArgumentTypeError("'batch_size' should be greater than 0!")

        if args.epochs is None or not args.epochs > 0:
            raise argparse.ArgumentTypeError("'epochs' should be greater than 0!")

        if args.learning_rate is None or not args.learning_rate > 0.:
            raise argparse.ArgumentTypeError("'learning_rate' should be greater than 0!")

        if args.momentum is None or not args.momentum > 0.:
            raise argparse.ArgumentTypeError("'momentum' should be greater than 0!")

        if args.weight_decay is None or not args.weight_decay > 0.:
            raise argparse.ArgumentTypeError("'weight_decay' should be greater than 0!")

        if args.poly_power is None or not args.poly_power > 0.:
            raise argparse.ArgumentTypeError("'poly_power' should be greater than 0!")
    else:
        if args.eval_file is not None:
            raise argparse.ArgumentTypeError("'eval_file' is required when 'train' parameter is false!")

        if not os.path.isfile(args.eval_file):
            raise argparse.ArgumentTypeError("File specified in 'eval_file' parameter doesn't exists!")

    main(train=args.train,
         eval_file=args.eval_file,
         device=('cuda' if args.device == 'gpu' else args.device),
         log_interval=args.log_interval,
         val_interval=args.val_interval,
         batch_size=args.batch_size,
         epochs=args.epochs,
         learning_rate=args.learning_rate,
         momentum=args.momentum,
         weight_decay=args.weight_decay,
         poly_power=args.poly_power,
         stage=args.stage,
         w1=args.w1,
         w2=args.w2)