import math
import torch as t
import torchvision as tv

import consts
from .ResNet101 import ResNet101
from .modules.ASPP import ASPP
from datasets.Cityscapes import settings as cityscapes_settings


class DSRLSS(t.nn.Module):
    MODEL_INPUT_SIZE = (512, 1024)
    MODEL_OUTPUT_SIZE = (1024, 2048)

    @staticmethod
    def _define_feature_extractor(num_classes, in_channels):
        # NOTE: Output size (B, 2048, 32, 64)
        feature_extractor_resnet101 = ResNet101(num_classes=num_classes,
                                                replace_stride_with_dilation=[False, False, True])

        # NOTE: Output size (B, 256, 32, 64)
        feature_summary_aspp = ASPP(in_channels=in_channels, out_channels=256, rate=1)
        
        return t.nn.ModuleDict({'backbone': feature_extractor_resnet101, 'aspp': feature_summary_aspp})

    @staticmethod
    def _define_SSSR_decoder(in_channels1, in_channels2, out_channels):
        decoder_modules = \
        {
            'upsample_sub': t.nn.Sequential(t.nn.Dropout(p=0.5),
                                            t.nn.UpsamplingBilinear2d(scale_factor=4)),
            'shortcut_conv': t.nn.Sequential(t.nn.Conv2d(in_channels=in_channels1,
                                                         out_channels=in_channels2,
                                                         kernel_size=1,
                                                         padding=0,
                                                         bias=True),
                                             t.nn.BatchNorm2d(num_features=in_channels2),
                                             t.nn.ReLU(inplace=True)),
            'cat_conv': t.nn.Sequential(t.nn.Conv2d(in_channels=(in_channels1+in_channels2),
                                                    out_channels=256,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                        t.nn.BatchNorm2d(num_features=256),
                                        t.nn.ReLU(inplace=True),
                                        t.nn.Dropout(p=0.5),
                                        t.nn.Conv2d(in_channels=256,
                                                    out_channels=256,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=True),
                                        t.nn.BatchNorm2d(num_features=256),
                                        t.nn.ReLU(inplace=True),
                                        t.nn.Dropout(p=0.1)),
            'cls_conv': t.nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1),
            # NOTE: Replaced this 'upsample4': t.nn.UpsamplingBilinear2d(scale_factor=4),
            # NOTE: Each 'ConvTranspose2d' scales 2x, so the following modules together scale by 8 times.
            'upsample16_pred': t.nn.Sequential(t.nn.ConvTranspose2d(in_channels=out_channels,
                                                                    out_channels=out_channels,
                                                                    kernel_size=2,
                                                                    stride=2,
                                                                    padding=0,
                                                                    bias=True),
                                               t.nn.BatchNorm2d(num_features=out_channels),
                                               t.nn.ReLU(inplace=True),
                                               t.nn.ConvTranspose2d(in_channels=out_channels,
                                                                    out_channels=out_channels,
                                                                    kernel_size=2,
                                                                    stride=2,
                                                                    padding=0,
                                                                    bias=True),
                                               t.nn.BatchNorm2d(num_features=out_channels),
                                               t.nn.ReLU(inplace=True),
                                               t.nn.ConvTranspose2d(in_channels=out_channels,
                                                                    out_channels=out_channels,
                                                                    kernel_size=2,
                                                                    stride=2,
                                                                    padding=0,
                                                                    bias=True),
                                               t.nn.BatchNorm2d(num_features=out_channels),
                                               t.nn.ReLU(inplace=True))
        }

        return t.nn.ModuleDict(decoder_modules)

    @staticmethod
    def _define_SISR_decoder(in_channels, out_channels, upscale_factor):
        return t.nn.Sequential(t.nn.Conv2d(in_channels=in_channels,
                                           out_channels=(out_channels * math.pow(upscale_factor, 2)),
                                           kernel_size=3,
                                           padding=1),
                               t.nn.PixelShuffle(upscale_factor=upscale_factor))

    @staticmethod
    def _define_feature_transformer(in_channels, out_channels):
        return t.nn.Sequential(t.nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           bias=True),
                               t.nn.BatchNorm2d(num_features=out_channels),
                               t.nn.ReLU(inplace=True))


    def __init__(self,
                 stage=None):
        super().__init__()

        # Save parameters to class instance variables
        self.stage = stage

        # Feature extractor
        self.feature_extractor = DSRLSS._define_feature_extractor(num_classes=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                  in_channels=2048)

        # Semantic Segmentation Super Resolution (SSSR)
        self.SSSR_decoder = DSRLSS._define_SSSR_decoder(in_channels1=256,
                                                        in_channels2=48,
                                                        out_channels=cityscapes_settings.DATASET_NUM_CLASSES)

        if self.train:
            if self.stage > 1:
                # Single Image Super-Resolution (SISR)
                self.SISR_decoder = DSRLSS._define_SISR_decoder(in_channels=48,
                                                                out_channels=consts.NUM_RGB_CHANNELS,
                                                                upscale_factor=4.0)

            if self.stage > 2:
                # Feature transform module for SSSR
                self.SSSR_feature_transformer = DSRLSS._define_feature_transformer(in_channels=cityscapes_settings.DATASET_NUM_CLASSES,
                                                                                   out_channels=1)

                # Feature transform module for SISR
                self.SISR_feature_transformer = DSRLSS._define_feature_transformer(in_channels=consts.NUM_RGB_CHANNELS,
                                                                                   out_channels=1)


    def forward(self, x):
        # Extract features
        backbone_features, lowlevel_features = self.feature_extractor['backbone'](x)
        aspp_features = self.feature_extractor['aspp'](backbone_features)
        upsample_sub = self.SSSR_decoder['upsample_sub'](aspp_features)
        shortcut_conv = self.SSSR_decoder['shortcut_conv'](lowlevel_features)
        cat_features = t.cat([upsample_sub, shortcut_conv], dim=1)

        # Semantic Segmentation Super Resolution (SSSR) decoder
        SSSR_output = self.SSSR_decoder['cat_conv'](cat_features)
        SSSR_output = self.SSSR_decoder['cls_conv'](SSSR_output)
        SSSR_output = self.SSSR_decoder['upsample16_pred'](SSSR_output)

        SISR_output = None
        if self.training:
            if self.stage > 1:
                # Single Image Super-Resolution (SISR) decoder
                SISR_output = self.SISR_decoder(cat_features)

                if self.stage > 2:
                    # Feature transform module for SSSR
                    SSSR_features_output = self.SSSR_feature_transformer(SSSR_output)

                    # Feature transform module for SISR
                    SISR_features_output = self.SISR_feature_transformer(SISR_output)

        return SSSR_output, SISR_output
