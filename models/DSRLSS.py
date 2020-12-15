import torch as t
import torchvision as tv

from .ResNet101 import ResNet101
from .modules.ASPP import ASPP
from datasets.Cityscapes import settings as cityscapes_settings


class DSRLSS(t.nn.Module):
    MODEL_INPUT_SIZE = (512, 1024)
    MODEL_OUTPUT_SIZE = (1024, 2048)

    @staticmethod
    def __define_feature_extractor():
        # NOTE: Output size (B, 2048, 32, 64)
        feature_extractor_resnet101 = ResNet101(num_classes=cityscapes_settings.DATASET_NUM_CLASSES,
                                                replace_stride_with_dilation=[False, False, True])

        # NOTE: Output size (B, 256, 32, 64)
        feature_summary_aspp = ASPP(in_channels=2048, out_channels=256, rate=1)
        
        return t.nn.ModuleDict({'backbone': feature_extractor_resnet101, 'aspp': feature_summary_aspp})

    @staticmethod
    def __define_SSSR_decoder():
        decoder_modules = \
        {
            'upsample_sub': t.nn.Sequential(t.nn.Dropout(p=0.5), t.nn.UpsamplingBilinear2d(scale_factor=4)),
            'shortcut_conv': t.nn.Sequential(t.nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1, padding=0, bias=True),
                                             t.nn.BatchNorm2d(num_features=48),
                                             t.nn.ReLU(inplace=True)),
            'cat_conv': t.nn.Sequential(t.nn.Conv2d(in_channels=(256+48), out_channels=256, kernel_size=3, padding=1, bias=True),
                                        t.nn.BatchNorm2d(num_features=256),
                                        t.nn.ReLU(inplace=True),
                                        t.nn.Dropout(p=0.5),
                                        t.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
                                        t.nn.BatchNorm2d(num_features=256),
                                        t.nn.ReLU(inplace=True),
                                        t.nn.Dropout(p=0.1)),
            'cls_conv': t.nn.Conv2d(in_channels=256, out_channels=cityscapes_settings.DATASET_NUM_CLASSES, kernel_size=1),
            #'upsample4': t.nn.UpsamplingBilinear2d(scale_factor=4),
            'upsample8_pred': t.nn.Sequential(t.nn.ConvTranspose2d(in_channels=cityscapes_settings.DATASET_NUM_CLASSES,     # Each ConvTranspose scales 2x
                                                                   out_channels=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                   kernel_size=2,
                                                                   stride=2,
                                                                   padding=0),
                                              t.nn.BatchNorm2d(num_features=cityscapes_settings.DATASET_NUM_CLASSES),
                                              t.nn.ReLU(inplace=True),
                                              t.nn.ConvTranspose2d(in_channels=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                   out_channels=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                   kernel_size=2,
                                                                   stride=2,
                                                                   padding=0),
                                              t.nn.BatchNorm2d(num_features=cityscapes_settings.DATASET_NUM_CLASSES),
                                              t.nn.ReLU(inplace=True),
                                              t.nn.ConvTranspose2d(in_channels=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                   out_channels=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                   kernel_size=2,
                                                                   stride=2,
                                                                   padding=0),
                                              t.nn.BatchNorm2d(num_features=cityscapes_settings.DATASET_NUM_CLASSES),
                                              t.nn.ReLU(inplace=True))
        }

        return t.nn.ModuleDict(decoder_modules)

    @staticmethod
    def __define_SISR_decoder():
        module_list = \
        [t.nn.Conv2d()]

        return t.nn.ModuleList(module_list)

    @staticmethod
    def __define_feature_transformer():
        module_list = \
        [t.nn.Conv2d()]

        return t.nn.ModuleList(module_list)


    def __init__(self,
                 stage=None):
        super().__init__()

        # Save parameters to class instance variables
        self.stage = stage

        # Feature extractor
        self.feature_extractor = DSRLSS.__define_feature_extractor()

        # Semantic Segmentation Super Resolution (SSSR)
        self.SSSR_decoder = DSRLSS.__define_SSSR_decoder()

        if self.train:
            if self.stage == 3:
                # Feature transform module for SSSR
                self.SSSR_feature_transform = DSRLSS.__define_feature_transformer()

            if self.stage in [2, 3]:
                # Single Image Super-Resolution (SISR)
                self.SISR_decoder = DSRLSS.__define_SISR_decoder(self.feature_extractor)

                if self.stage == 3:
                    # Feature transform module for SISR
                    self.SISR_feature_transform = DSRLSS.__define_feature_transformer()

    def forward(self, x):
        # Extract features
        backbone_features, lowlevel_features = self.feature_extractor['backbone'](x)
        aspp_features = self.feature_extractor['aspp'](backbone_features)

        # Semantic Segmentation Super Resolution (SSSR)
        upsample_sub = self.SSSR_decoder['upsample_sub'](aspp_features)
        shortcut_conv = self.SSSR_decoder['shortcut_conv'](lowlevel_features)
        SSSR_output = t.cat([upsample_sub, shortcut_conv], dim=1)
        SSSR_output = self.SSSR_decoder['cat_conv'](SSSR_output)
        SSSR_output = self.SSSR_decoder['cls_conv'](SSSR_output)
        SSSR_output = self.SSSR_decoder['upsample8_pred'](SSSR_output)

        SISR_output = None
        if self.training:
            if self.stage == 3:
                # Feature transform module for SSSR
                SSSR_output = self.SSSR_feature_transform(SSSR_output)

            if self.stage in [2, 3]:
                # Single Image Super-Resolution (SISR)
                SISR_output = self.SISR_decoder(features)

                if self.stage == 3:
                    # Feature transform module for SISR
                    SISR_output = self.SISR_feature_transform(SISR_output)

        return SSSR_output, SISR_output
