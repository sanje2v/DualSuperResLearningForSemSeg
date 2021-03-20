import sys
import os
import os.path
import shutil
import functools
import argparse
import multiprocessing
import json
import numpy as np
import torch as t
import torchvision as tv

from models import DSRL
import command_handlers
from utils import *
import settings


# NOTE: Entry function for distributed spawned workers
def distributed_main(device_id, args):
    # NOTE: args['distributed'] = ['MASTER_ADDR', 'MASTER_PORT', 'NODES', 'DEVICES_PER_NODE', 'BACKEND', 'INIT_METHOD', 'NODE_ID', 'DEVICE_ID', 'WORLD_SIZE', 'RANK']
    args = json.loads(args)
    args['distributed'] =\
    {
        'MASTER_ADDR': args['distributed'][0],
        'MASTER_PORT': args['distributed'][1],
        'NODES': args['distributed'][2],
        'DEVICES_PER_NODE': args['distributed'][3],
        'BACKEND': args['distributed'][4],
        'INIT_METHOD': args['distributed'][5],
        'NODE_ID': args['distributed'][6],
        'DEVICE_ID': device_id,
        'WORLD_SIZE': args['distributed'][2] * args['distributed'][3],      # NODES * DEVICES_PER_NODE
        'RANK': args['distributed'][6] * args['distributed'][3] + device_id # NODE_ID * DEVICES_PER_NODE + DEVICE_ID
    }

    os.environ['MASTER_ADDR'] = args['distributed']['MASTER_ADDR']
    os.environ['MASTER_PORT'] = str(args['distributed']['MASTER_PORT'])    # CAUTION: Cannot assign integer to environmental variable, hence the 'str()'
    os.environ['WORLD_SIZE'] = str(args['distributed']['WORLD_SIZE'])
    os.environ['RANK'] = str(args['distributed']['RANK'])

    print(INFO("Rank {:d} worker started.".format(args['distributed']['RANK'])))

    main(args)


def main(args):
    # Load variables from checkpoint if resuming training
    if args['command'] == 'resume-train':
        checkpoint_dict = load_checkpoint_or_weights(args['checkpoint'])

        for variable in settings.VARIABLES_IN_CHECKPOINT:
            args[variable] = checkpoint_dict[variable]

    if 'disable_cudnn_benchmark' in args:
        t.backends.cudnn.benchmark = not args['disable_cudnn_benchmark']

    # If there is 'dataset' key specified, add dataset's class, name, split and starting index
    if 'dataset' in args:
        if isinstance(args['dataset'], str):
            args['dataset'] = [args['dataset'], 'train', 0]

        dataset_dict = dict(settings.DATASETS[args['dataset'][0]])  # NOTE: Create deep copy
        for i, item in enumerate(args['dataset']):
            if i == 0:
                dataset_dict['name'] = item
            elif i == 1:
                dataset_dict['split'] = item
            elif i == 2:
                dataset_dict['starting_index'] = item
        args['dataset'] = dataset_dict

    # According to 'args.command' call functions in 'commands' module
    if args['command'] in ['train', 'resume-train']:
        train_logs_dir = os.path.join(args['experiment_id'], settings.LOGS_DIR.format(stage=args['stage'], mode='train'))
        os.makedirs(train_logs_dir, exist_ok=True)

        # All calls to 'print()' is to be redirected to 'tqdm.write()' and a file
        with OverridePrintWithTQDMWriteAndLog(os.path.join(train_logs_dir, settings.STDOUT_FILE)) as stdout:
            try:
                args['is_resuming_training'] = (args['command'] == 'resume-train')
                command_handlers.train_or_resume(**args)

            except KeyboardInterrupt as ex:
                stdout.write("Caught Ctrl+c to interrupt training!")
                raise ex

            except Exception as ex:
                stdout.write("Exception caught: {}".format(str(ex)))
                raise ex
    else:
        with OverridePrintWithTQDMWriteAndLog():  # All calls to 'print()' is to be redirected to 'tqdm.write()'
            # CAUTION: 'argparse' library will create variable for commandline option with '-' converted to '_'.
            #   Hence, the following 'replace()' is required.
            command_func_to_call = getattr(command_handlers, args['command'].replace('-', '_'), None)
            assert command_func_to_call, "BUG CHECK: Command '{:s}' does not have any implementation under 'command_handlers' package.".format(args['command'])
            command_func_to_call(**args)



def parse_cmdline_and_invoke_main(args):
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
        train_parser.add_argument('--device', default=settings.DEFAULT_DEVICE, type=str.casefold, choices=settings.SUPPORTED_DEVICES, help="Device to create model in, cpu/gpu")
        train_parser.add_argument('--distributed', required=False, nargs=7, metavar=('MASTER_ADDR', 'MASTER_PORT', 'NODES', 'DEVICES_PER_NODE', 'BACKEND', 'INIT_METHOD', 'NODE_ID'), const=settings.SUPPORTED_DISTRIBUTED_BACKENDS, action=ValidateDistributedTrainingOptions, help="Enable distributed training")
        train_parser.add_argument('--mixed-precision', default=settings.DEFAULT_AMP_OPTIMIZATION_OPTION, type=str.upper, choices=settings.AMP_OPTIMIZATION_OPTIONS, help="Enable mixed precision training with the ability to mixed float16 and 32 using apex optimization flags")
        train_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make training slower")
        train_parser.add_argument('--profile', action='store_true', help="Enable PyTorch profiling of execution times and memory usage")
        train_parser.add_argument('--num-workers', default=settings.DEFAULT_NUM_WORKERS, type=int, help="No. of workers for data loader")
        train_parser.add_argument('--dataset', required=True, type=str.casefold, choices=settings.DATASETS.keys(), help="Dataset to operate on")
        train_parser.add_argument('--val-interval', default=settings.DEFAULT_VAL_INTERVAL, type=int, help="Epoch intervals after which to perform validation")
        train_parser.add_argument('--checkpoint-interval', default=settings.DEFAULT_CHECKPOINT_INTERVAL, type=int, help="Epoch intervals to create checkpoint after in training")
        train_parser.add_argument('--checkpoint-history', default=settings.DEFAULT_CHECKPOINT_HISTORY, type=int, help="No. of latest autosaved checkpoints to keep while deleting old ones, 0 to disable autosave")
        train_parser.add_argument('--init-weights', default=None, type=str, help="Load initial weights file for model")
        train_parser.add_argument('--batch-size', default=settings.DEFAULT_BATCH_SIZE, type=int, help="Batch size to use for training and testing")
        train_parser.add_argument('--epochs', required=True, type=int, help="No. of epochs to train")
        train_parser.add_argument('--learning-rate', type=float, default=settings.DEFAULT_LEARNING_RATE, help="Learning rate to begin training with")
        train_parser.add_argument('--end-learning-rate', type=float, default=settings.DEFAULT_END_LEARNING_RATE, help="End learning rate for the last epoch")
        train_parser.add_argument('--momentum', type=float, default=settings.DEFAULT_MOMENTUM, help="Momentum value for SGD")
        train_parser.add_argument('--weights-decay', type=float, default=settings.DEFAULT_WEIGHTS_DECAY, help="Weights decay for SGD")
        train_parser.add_argument('--poly-power', type=float, default=settings.DEFAULT_POLY_POWER, help="Power for poly learning rate strategy")
        train_parser.add_argument('--stage', required=True, type=int, choices=settings.STAGES, help="0: Train SSSR only\n1: Train SSSR+SISR\n2: Train SSSR+SISR with feature affinity")
        train_parser.add_argument('--w1', type=float, default=settings.DEFAULT_LOSS_WEIGHTS[0], help="Weight for MSE loss")
        train_parser.add_argument('--w2', type=float, default=settings.DEFAULT_LOSS_WEIGHTS[1], help="Weight for FA loss")
        train_parser.add_argument('--freeze-batch-norm', action='store_true', help="Keep all Batch Normalization layers disabled while training")
        train_parser.add_argument('--experiment-id', type=str, default='', help="Experiment ID which is used to create a root directory for weights and logs directories")
        train_parser.add_argument('--description', type=str, default=None, help="Description of experiment to be saved in 'params.txt' with given commandline parameters")
        train_parser.add_argument('--early-stopping', action='store_true', help="Automatically stop training when training error is less than validation error")
        train_parser.add_argument('--dry-run', action='store_true', help="Disable actual training and validation code used to debug boilerplate code around them")

        # Training with configuration from JSON file
        config_train_parser = command_parser.add_parser('config-train', help="JSON configuration file that provides commandline parameters for training")
        config_train_parser.add_argument('--file', required=True, type=str, help="Path to JSON configuration file")

        # Resume training from checkpoint arguments
        resume_train_parser = command_parser.add_parser('resume-train', help="Resume training model from checkpoint file")
        resume_train_parser.add_argument('--checkpoint', required=True, type=str, help="Resume training with given checkpoint file")
        resume_train_parser.add_argument('--distributed', required=False, nargs=7, metavar=('MASTER_ADDR', 'MASTER_PORT', 'NODES', 'DEVICES_PER_NODE', 'BACKEND', 'INIT_METHOD', 'NODE_ID'), const=settings.SUPPORTED_DISTRIBUTED_BACKENDS, action=ValidateDistributedTrainingOptions, help="Enable distributed training")
        resume_train_parser.add_argument('--dataset', required=True, type=str.casefold, choices=settings.DATASETS.keys(), help="Dataset to operate on")

        # Evaluation arguments
        test_parser = command_parser.add_parser('test', help="Test trained weights with a single input image")
        test_source = test_parser.add_mutually_exclusive_group(required=True)
        test_source.add_argument('--image-file', type=str, help="Run evaluation on a image file using trained weights")
        test_source.add_argument('--images-dir', type=str, help="Run evaluation on image files (JPG and PNG) in specified directory")
        test_source.add_argument('--dataset', nargs=3, metavar=('DATASET', 'SPLIT', 'STARTING_INDEX'), const=settings.DATASETS, action=ValidateDatasetNameSplitAndIndex, help="Dataset, split and starting index to test from")
        test_parser.add_argument('--output-dir', type=str, default=settings.OUTPUTS_DIR, help="Specify directory where testing results are saved")
        test_parser.add_argument('--weights', required=True, type=str, help="Weights file to use")
        test_parser.add_argument('--device', default=settings.DEFAULT_DEVICE, type=str.casefold, choices=settings.SUPPORTED_DEVICES, help="Device to create model in, cpu/gpu")
        test_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make evaluation slower")
        test_parser.add_argument('--profile', action='store_true', help="Enable PyTorch profiling of execution times and memory usage")
        test_parser.add_argument('--compiled-model', action='store_true', help="Using compiled model in '--weights' made using 'compile-model' command")

        # Purge weights and logs
        purge_weights_logs = command_parser.add_parser('purge-weights-logs', help="Delete all training weights and logs")
        purge_weights_logs_type = purge_weights_logs.add_mutually_exclusive_group(required=True)
        purge_weights_logs_type.add_argument('--stage', type=int, choices=settings.STAGES, help="Stage for which to delete weights and logs")
        purge_weights_logs_type.add_argument('--all', action='store_true', help="Delete weights and logs for all stages")

        # Print model arguments
        print_model_parser = command_parser.add_parser('print-model', help="Prints all the layers in the model with extra information for a stage")
        print_model_parser.add_argument('--stage', required=True, type=int, choices=settings.STAGES, help="Stage to print layers of model for")
        print_model_parser.add_argument('--dataset', type=str.casefold, choices=settings.DATASETS.keys(), default=list(settings.DATASETS.keys())[0], help="Dataset settings to use")

        # Purne weights arguments
        purne_weights_parser = command_parser.add_parser('purne-weights', help="Removes all weights from a weights file which are not needed for inference")
        purne_weights_parser.add_argument('--src-weights', required=True, type=str, help="Checkpoint/Weights file to prune")
        purne_weights_parser.add_argument('--dest-weights', required=True, type=str, help="New weights file to write to")
        purne_weights_parser.add_argument('--dataset', type=str.casefold, choices=settings.DATASETS.keys(), default=list(settings.DATASETS.keys())[0], help="Dataset settings to use")

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
        benchmark_parser.add_argument('--weights', required=True, type=str, help="Weights to use")
        benchmark_parser.add_argument('--dataset', required=True, nargs=2, metavar=('DATASET', 'SPLIT'), action=ValidateDatasetNameAndSplit, const=settings.DATASETS, help="Dataset and split to operate on")
        benchmark_parser.add_argument('--device', default=settings.DEFAULT_DEVICE, type=str.casefold, choices=settings.SUPPORTED_DEVICES, help="Device to create model in, cpu/gpu")
        benchmark_parser.add_argument('--disable-cudnn-benchmark', action='store_true', help="Disable CUDNN benchmark mode which might make training slower")
        benchmark_parser.add_argument('--num-workers', default=settings.DEFAULT_NUM_WORKERS, type=int, help="Number of workers for data loader")
        benchmark_parser.add_argument('--batch-size', default=settings.DEFAULT_BATCH_SIZE, type=int, help="Batch size to use for benchmarking")

        # Compile model arguments
        compile_model_parser = command_parser.add_parser('compile-model', help="Compiles given model using TorchScript and outputs a compiled file")
        compile_model_parser.add_argument('--weights', required=True, type=str, help="Weights to use")
        compile_model_parser.add_argument('--output-file', required=True, type=str, help="Output file to compile the model to")
        compile_model_parser.add_argument('--dataset', type=str.casefold, choices=settings.DATASETS.keys(), default=list(settings.DATASETS.keys())[0], help="Dataset settings to use")


        # Validate arguments according to mode
        args = parser.parse_args(args)
        if args.command == 'train':
            if args.distributed:
                if not t.distributed.is_available():
                    raise argparse.ArgumentTypeError("Installed version of PyTorch is not compiled with distributed training support!")

                if not isCUDAdevice(args.device):
                    raise argparse.ArgumentTypeError("'--distributed' option cannot be used with non-CUDA device specified in '--device'!")

            if isCUDAdevice(args.device) and not t.cuda.is_available():
                raise Exception("CUDA is not available to use for accelerated computing!")

            if not isCUDAdevice(args.device):
                if args.disable_cudnn_benchmark:
                    raise argparse.ArgumentTypeError("'--disable-cudnn-benchmark' is unsupported in non-CUDA devices!")

                if args.mixed_precision:
                    raise argparse.ArgumentTypeError("'--device' specified must be a CUDA device when specifying '--mixed-precision'!")

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

                # CAUTION: Some functions might fail for relative paths so we convert them to absolute path
                args.init_weights = os.path.abspath(args.init_weights)

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

            if args.experiment_id:
                if isInvalidFilename(args.experiment_id):
                    raise argparse.ArgumentTypeError("'--experiment-id' must not contain invalid filename characters ({:s})!".format(', '.join(INVALID_FILENAME_CHARS)))

                args.experiment_id = os.path.join(settings.EXPERIMENTS_ROOT_DIR, args.experiment_id)
                if os.path.isdir(args.experiment_id):
                    raise argparse.ArgumentTypeError("'--experiment-id' already exists and overwriting experiment directory is not supported!")

            # Warning if there are already weights for this stage
            if os.path.isfile(os.path.join(args.experiment_id, settings.WEIGHTS_DIR.format(stage=args.stage), settings.FINAL_WEIGHTS_FILE)):
                answer = input(CAUTION("Weights file for this stage already exists. Training will delete the current weights and logs. Continue? (y/n) ")).casefold()
                if answer == 'y':
                    shutil.rmtree(os.path.join(args.experiment_id, settings.LOGS_DIR.format(stage=args.stage, mode='')), ignore_errors=True)
                    shutil.rmtree(os.path.join(args.experiment_id, settings.WEIGHTS_DIR.format(stage=args.stage)))
                else:
                    sys.exit(0)

        elif args.command == 'config-train':
            if not os.path.isfile(args.file):
                raise argparse.ArgumentTypeError("File specified in '--file' parameter doesn't exists!")

            # Load configuration file, spawn a separate process with commandline parameters in this file and wait for child process to exit
            try:
                def correct_JSON_parse_hook(pairs):
                    # Prepend a '--' as required for subcommand and convert its arguments to string
                    return {('--' + c): str(a) for c, a in pairs}

                with open(args.file, 'r') as train_config_file:
                    train_config_dict = json.load(train_config_file, object_pairs_hook=correct_JSON_parse_hook)

                # Create list of commandline args to send to child process
                train_process_args = ['train', *functools.reduce(lambda k, v: k + v, train_config_dict.items())]
                train_process = multiprocessing.Process(target=parse_cmdline_and_invoke_main,
                                                        args=(train_process_args,))
                train_process.start()
                train_process.join()
                sys.exit(train_process.exitcode)

            except json.JSONDecodeError as ex:
                raise argparse.ArgumentTypeError("Parsing configuration JSON file raised exception: {:}".format(str(ex)))
                sys.exit(-1)

            except KeyboardInterrupt:
                sys.exit(0)

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

            if not any(hasExtension(args.weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.weights):
                raise argparse.ArgumentTypeError("Couldn't find weights file '{:s}'!".format(args.weights))

            if isCUDAdevice(args.device) and not t.cuda.is_available():
                raise Exception("CUDA is not available to use for accelerated computing!")

            if not isCUDAdevice(args.device) and args.disable_cudnn_benchmark:
                raise argparse.ArgumentTypeError("'--disable-cudnn-benchmark' is unsupported in non-CUDA devices!")

        elif args.command == 'purge-weights-logs':
            answer = input('This will delete {:s} logs and weights. Continue? (y/n) '.format('all' if args.all else 'stage {:d}'.format(args.stage)))
            if answer.casefold() == 'y':
                purge_start_stage = settings.STAGES[0] if args.all else args.stage
                purge_stop_stage = settings.STAGES[-1] if args.all else args.stage

                for stage in range(purge_start_stage, purge_stop_stage+1):
                    logs_dir = settings.LOGS_DIR.format(stage=stage, mode='')
                    weights_dir = settings.WEIGHTS_DIR.format(stage=stage)

                    for dir in [logs_dir, weights_dir]:
                        if os.path.isdir(dir):
                            shutil.rmtree(dir)
            sys.exit(0)

        elif args.command == 'purne-weights':
            if not any(hasExtension(args.src_weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--src-weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.src_weights):
                raise argparse.ArgumentTypeError("File specified in '--src-weights' parameter doesn't exists!")

            if os.path.isfile(args.dest_weights):
                answer = input(CAUTION("Destination weights file specified already exists. This will overwrite the file. Continue (y/n)? ")).casefold()
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

            if isCUDAdevice(args.device) and not t.cuda.is_available():
                raise Exception("CUDA is not available to use for accelerated computing!")

            if not isCUDAdevice(args.device) and args.disable_cudnn_benchmark:
                raise argparse.ArgumentTypeError("'--disable-cudnn-benchmark' is unsupported in non-CUDA devices!")

            if not args.num_workers >= 0:
                raise argparse.ArgumentTypeError("'--num-workers' should be greater than or equal to 0!")

            if not args.batch_size > 0:
                raise argparse.ArgumentTypeError("'--batch-size' should be greater than 0!")

        elif args.command == 'compile-model':
            if not any(hasExtension(args.weights, x) for x in ['.checkpoint', '.weights']):
                raise argparse.ArgumentTypeError("'--weights' must be of either '.checkpoint' or '.weights' file type!")

            if not os.path.isfile(args.weights):
                raise argparse.ArgumentTypeError("Couldn't find weights file '{:s}'!".format(args.weights))


        with t.autograd.profiler.profile(enabled=getattr(args, 'profile', False),
                                         with_stack=True,
                                         use_cuda=hasattr(args, 'device') and isCUDAdevice(args.device),
                                         profile_memory=True) as profiler:
            # Do action in 'command'
            if getattr(args, 'distributed', None):
                t.multiprocessing.spawn(distributed_main, args=(json.dumps(args.__dict__),), nprocs=args.distributed[3], join=True)
            else:
                main(args.__dict__)


    except KeyboardInterrupt:
        print(CAUTION("Caught 'Ctrl+c' SIGINT signal. Aborted operation."))

    except argparse.ArgumentTypeError as ex:
        print(FATAL("{:s}\n".format(str(ex))))
        parser.print_usage()

    finally:
        # If a profiler is active, stop it and save results to disk
        if profiler:
            profiling_filename = os.path.join(settings.OUTPUTS_DIR, settings.PROFILING_FILE)
            profiler.export_chrome_trace(profiling_filename)
            print(INFO("Profiling output has been saved to '{:s}'.".format(profiling_filename)))

        if t.distributed.is_initialized():
            t.distributed.destroy_process_group()


if __name__ == '__main__':
    parse_cmdline_and_invoke_main(sys.argv[1:])