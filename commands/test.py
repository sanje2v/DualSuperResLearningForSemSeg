import os
import os.path
from tqdm.auto import tqdm
import numpy as np
import torch as t
import torchvision as tv
from PIL import Image, ImageOps

from models import DSRL
from models.transforms import *
from utils import *
import settings
import consts
from datasets.Cityscapes import settings as cityscapes_settings


def test(image_file, images_dir, dataset, output_dir, weights, device, device_obj, **other_args):
    # Testing on a single input image using given weights

    # Create model and set to evaluation mode
    model = DSRL(stage=1).eval()

    # Load specified weights file
    model.load_state_dict(load_checkpoint_or_weights(weights, map_location=device_obj)['model_state_dict'], strict=True)

    # Copy the model into 'device_obj'
    model = model.to(device_obj)

    if image_file or images_dir:
        image_filenames = [image_file,] if image_file else getFilesWithExtension(images_dir, consts.IMAGE_FILE_EXTENSIONS, with_path=True)

        for image_filename in tqdm(image_filenames,
                                   desc='TESTING',
                                   colour='yellow',
                                   position=0,
                                   leave=False):
            # Using an image file for testing
            # Load image file, rotate according to EXIF info, add 'batch' dimension and convert to tensor
            with ImageOps.exif_transpose(Image.open(image_filename))\
                    .convert('RGB')\
                    .resize(swapTupleValues(DSRL.MODEL_OUTPUT_SIZE), resample=Image.BILINEAR) as input_image:
                with timethis(INFO("Inference required {:}.")), t.no_grad():
                    input_transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                             tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD),
                                                             tv.transforms.Resize(size=DSRL.MODEL_INPUT_SIZE, interpolation=Image.BILINEAR),
                                                             tv.transforms.Lambda(lambda x: t.unsqueeze(x, dim=0))])
                    SSSR_output, _, _, _ = model.forward(input_transform(input_image).to(device_obj))

                input_image = np.array(input_image, dtype=np.uint8).transpose((2, 0, 1))
                SSSR_output = np.argmax(np.squeeze(SSSR_output.detach().cpu().numpy(), axis=0), axis=0)    # Bring back result to CPU memory and convert to index array
                vis_image = make_input_output_visualization(input_image, SSSR_output, cityscapes_settings.CLASS_RGB_COLOR)
                vis_image = vis_image.transpose((1, 2, 0))    # Channel order required for PIL.Image below

            with Image.fromarray(vis_image, mode='RGB') as vis_image:    # Convert from numpy array to PIL Image
                # Save and show output on plot
                os.makedirs(output_dir, exist_ok=True)
                vis_image_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_filename))[0] + '.png')
                vis_image.save(vis_image_filename, format='PNG')
                # Only show output if a single image file is specified
                if image_file:
                    vis_image.show(title='Segmentation output')

            tqdm.write(INFO("Output image saved as: {0:s}.".format(vis_image_filename)))
    else:
        dataset_split, starting_index = dataset

        joint_transforms = JointCompose([JointImageAndLabelTensor(cityscapes_settings.LABEL_MAPPING_DICT),
                                         lambda img, seg: (tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD)(img), seg),
                                         lambda img, seg: (DuplicateToScaledImageTransform(new_size=DSRL.MODEL_INPUT_SIZE)(img), seg)])
        dataset = tv.datasets.Cityscapes(settings.CITYSCAPES_DATASET_DATA_DIR,
                                         split=dataset_split,
                                         mode='fine',
                                         target_type='semantic',
                                         transforms=joint_transforms)
        loader = t.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=isCUDAdevice(device),
                                         drop_last=False)

        with t.no_grad():
            tqdm.write(INFO("Press ENTER to show next pair of input and output. Use CTRL+c to quit."))
            for i, ((input_scaled, input_org), target) in enumerate(tqdm(loader,
                                                                         desc='TESTING',
                                                                         colour='yellow',
                                                                         position=0,
                                                                         leave=False)):
                if i >= starting_index:
                    with timethis(INFO("Inference required {:}.")), t.no_grad():
                        SSSR_output, _, _, _ = model.forward(input_scaled.to(device_obj))

                    input_image = input_org.detach().cpu().numpy()[0]
                    input_image = np.array(cityscapes_settings.DATASET_STD).reshape(consts.NUM_RGB_CHANNELS, 1, 1) * input_image +\
                                  np.array(cityscapes_settings.DATASET_MEAN).reshape(consts.NUM_RGB_CHANNELS, 1, 1)
                    input_image = np.clip(input_image * 255., a_min=0.0, a_max=255.).astype(np.uint8)
                    SSSR_output = np.argmax(SSSR_output.detach().cpu().numpy()[0], axis=0)    # Bring back result to CPU memory and convert to index array
                    target = target.detach().cpu().numpy()[0]
                    SSSR_output[target == cityscapes_settings.IGNORE_CLASS_LABEL] = cityscapes_settings.IGNORE_CLASS_LABEL
                    vis_image_target = make_input_output_visualization(input_image, target, cityscapes_settings.CLASS_RGB_COLOR)
                    vis_image_pred = make_input_output_visualization(input_image, SSSR_output, cityscapes_settings.CLASS_RGB_COLOR)
                    vis_image = np.concatenate((vis_image_target, vis_image_pred), axis=1)
                    vis_image = vis_image.transpose((1, 2, 0))    # Channel order required for PIL.Image below

                    with Image.fromarray(vis_image, mode='RGB') as vis_image:    # Convert from numpy array to PIL Image
                        # Save and show output on plot
                        os.makedirs(output_dir, exist_ok=True)
                        vis_image_filename = os.path.join(output_dir, str(i) + '.png')
                        vis_image.save(vis_image_filename, format='PNG')
                        vis_image.show(title='Segmentation output')
                    tqdm.write(INFO("Output image saved as: {0:s}.".format(vis_image_filename)))

                    input()