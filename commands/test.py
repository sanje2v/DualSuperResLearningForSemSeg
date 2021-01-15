import os
import os.path
from tqdm.auto import tqdm as tqdm
import numpy as np
import torch as t
import torchvision as tv
from PIL import Image, ImageOps

from models import DSRL
from utils import *
import consts
import settings
from datasets.Cityscapes import settings as cityscapes_settings


def test(image_file, weights, device, device_obj, **other_args):
    # Testing on a single input image using given weights

    # Create model and set to evaluation mode disabling all batch normalization layers
    model = DSRL(stage=1).eval()

    # Load specified weights file
    model.load_state_dict(load_checkpoint_or_weights(weights, map_location=device_obj)['model_state_dict'], strict=True)

    # Copy the model into 'device_obj'
    model = model.to(device_obj)

    # Load image file, rotate according to EXIF info, add 'batch' dimension and convert to tensor
    with ImageOps.exif_transpose(Image.open(image_file))\
            .convert('RGB')\
            .resize(swapTupleValues(DSRL.MODEL_OUTPUT_SIZE), resample=Image.BILINEAR) as input_image:
        with timethis(INFO("Inference required {:}.")), t.no_grad():
            input_transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=cityscapes_settings.DATASET_MEAN, std=cityscapes_settings.DATASET_STD),
                                                     tv.transforms.Resize(size=DSRL.MODEL_INPUT_SIZE, interpolation=Image.BILINEAR),
                                                     tv.transforms.Lambda(lambda x: t.unsqueeze(x, dim=0))])
            SSSR_output, _, _, _ = model.forward(input_transform(input_image).to(device_obj))
            SSSR_output = np.squeeze(SSSR_output.detach().cpu().numpy(), axis=0)    # Bring back result to CPU memory and remove batch dimension

        output_image = make_output_visualization(SSSR_output, input_image, DSRL.MODEL_OUTPUT_SIZE, cityscapes_settings.CLASS_RGB_COLOR)

    with Image.fromarray(output_image, mode='RGB') as output_image:    # Convert from numpy array to PIL Image
        # Save and show output on plot
        os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
        output_image_filename = os.path.join(settings.OUTPUTS_DIR, os.path.splitext(os.path.basename(image_file))[0] + '.png')

        output_image.save(output_image_filename, format='PNG')
        output_image.show(title='Segmentation output')

    tqdm.write(INFO("Output image saved as: {0:s}.".format(output_image_filename)))