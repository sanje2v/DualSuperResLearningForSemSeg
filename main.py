import sys
import os
import os.path
import shutil
import argparse
from tqdm.auto import tqdm as tqdm
import numpy as np
import torch as t
import torchvision as tv

import commands
from utils import *
import settings



def main(profiler, **args):
    # Load variables from checkpoint if resuming training
    if args['command'] == 'resume-train':
        checkpoint_dict = load_checkpoint_or_weights(args['checkpoint'])

        for variable in settings.VARIABLES_IN_CHECKPOINT:
            args[variable] = checkpoint_dict[variable]

    # Check and prepare device, if specified
    if 'device' in args:
        args['device_obj'] = t.device('cuda' if args['device'] == 'gpu' else args['device'])

        # Device to perform calculation in
        if isCUDAdevice(args['device']):
           if not t.cuda.is_available():
               raise Exception("CUDA is not available to use for accelerated computing!")

           if 'disable_cudnn_benchmark' in args:
               t.backends.cudnn.benchmark = not args['disable_cudnn_benchmark']

    # According to 'args.command' call functions in 'commands' module
    if args['command'] in ['train', 'resume-train']:
        commands.train_or_resume(**args)
    else:
        # CAUTION: 'argparse' library will create variable for commandline option with '-' converted to '_'.
        #   Hence, the following 'replace()' is required.
        command_func_to_call = getattr(commands, args['command'].replace('-', '_'), None)
        assert command_func_to_call, "BUG CHECK: Command '{:s}' does not have any implementation under 'commands' package.".format(args['command'])
        command_func_to_call(**args)



if __name__ == '__main__':
    assert check_version(sys.version_info, *settings.MIN_PYTHON_VERSION), \
        FATAL("This program needs at least Python {0:d}.{1:d} interpreter.".format(*settings.MIN_PYTHON_VERSION))
    assert check_version(t.__version__, *settings.MIN_PYTORCH_VERSION), \
        FATAL("This program needs at least PyTorch {0:d}.{1:d}.".format(*settings.MIN_PYTORCH_VERSION))
    assert check_version(tv.__version__, *settings.MIN_TORCHVISION_VERSION), \
        FATAL("This program needs at least TorchVision {0:d}.{1:d}.".format(*settings.MIN_TORCHVISION_VERSION))
    assert check_version(np.__version__, *settings.MIN_NUMPY_VERSION), \
        FATAL("This program needs at least NumPy {0:d}.{1:d}.".format(*settings.MIN_NUMPY_VERSION))

    profiler = None
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
        test_source = test_parser.add_mutually_exclusive_group(required=True)
        test_source.add_argument('--image-file', type=str, help="Run evaluation on a image file using trained weights")
        test_source.add_argument('--images-dir', type=str, help="Run evaluation on image files (JPG and PNG) in specified directory")
        test_source.add_argument('--dataset', type=lambda x: int(x) if x.isnumeric() else x, nargs=2, metavar=('SPLIT', 'STARTING_INDEX'), default='test 0', help="Run evaluation on dataset split starting from specified index")
        test_parser.add_argument('--output-dir', type=str, default=settings.OUTPUTS_DIR, help="Specify directory where testing results are saved")
        test_parser.add_argument('--weights', type=str, required=True, help="Weights file to use")
        test_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        test_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make evaluation slower")
        test_parser.add_argument('--profile', action='store_true', help="Enable PyTorch profiling of execution times and memory usage")

        # Print model arguments
        print_model_parser = command_parser.add_parser('print-model', help="Prints all the layers in the model with extra information for a stage")
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
        benchmark_parser.add_argument('--dataset-split', type=str.lower, choices=settings.DATASET_SPLITS, default='test', help="Which dataset's split to benchmark")
        benchmark_parser.add_argument('--device', default='gpu', type=str.lower, help="Device to create model in, cpu/gpu/cuda:XX")
        benchmark_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make training slower")
        benchmark_parser.add_argument('--num-workers', default=4, type=int, help="Number of workers for data loader")
        benchmark_parser.add_argument('--batch-size', default=6, type=int, help="Batch size to use for benchmarking")


        # Validate arguments according to mode
        args = parser.parse_args()
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
            if args.image_file and not os.path.isfile(args.image_file):
                raise argparse.ArgumentTypeError("File specified in '--image-file' parameter doesn't exists!")

            if args.images_dir and not os.path.isdir(args.images_dir):
                raise argparse.ArgumentTypeError("Directory specified in '--images-dir' parameter doesn't exists!")

            if args.dataset and not args.dataset[0] in settings.DATASET_SPLITS:
                raise argparse.ArgumentTypeError("Dataset split must be one of {:s}!".format(", ".join(settings.DATASET_SPLITS)))

            if args.dataset and not type(args.dataset[1]) is int:
                raise argparse.ArgumentTypeError("Dataset starting index must be an integer that is equal or greater than 0!")

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

        with t.autograd.profiler.profile(enabled=getattr(args, 'profile', False),
                                         use_cuda=hasattr(args, 'device') and isCUDAdevice(args.device),
                                         record_shapes=True,
                                         profile_memory=True) as profiler:
            # Do action in 'command'
            main(profiler, **args.__dict__)


    except KeyboardInterrupt:
        tqdm.write(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))

    except argparse.ArgumentTypeError as ex:
        tqdm.write(FATAL("{:s}".format(str(ex))))
        parser.print_usage()

    finally:
        # If a profiler is active, stop it and save results to disk
        if profiler:
            profiling_filename = os.path.join(settings.OUTPUTS_DIR, settings.PROFILING_FILE)
            profiler.export_chrome_trace(profiling_filename)
            tqdm.write(INFO("Profiling output has been saved to '{:s}'.".format(profiling_filename)))