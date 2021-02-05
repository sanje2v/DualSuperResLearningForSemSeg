import torch as t
import torchvision as tv

import consts
from .modules.backbone import ResNet101
from .modules.ASPP import ASPP

from .BaseModel import BaseModel


class DSRL(BaseModel):
    STAGES = [1, 2, 3]
    MODEL_INPUT_SIZE = (512, 1024)
    MODEL_OUTPUT_SIZE = (1024, 2048)

    @staticmethod
    def _define_feature_extractor(in_channels:int, out_channels1:int, out_channels2:int):
        feature_extractor_modules = \
        {
            'backbone': ResNet101(replace_stride_with_dilation=[False, False, True]),
            'aspp': ASPP(in_channels=in_channels, out_channels=out_channels1, rate=1),
            'upsample_sub': t.nn.Sequential(t.nn.Dropout(p=0.5),
                                            t.nn.UpsamplingBilinear2d(scale_factor=4.0)),
            'shortcut_conv': t.nn.Sequential(t.nn.Conv2d(in_channels=out_channels1,
                                                         out_channels=out_channels2,
                                                         kernel_size=1,
                                                         padding=0,
                                                         bias=False),
                                             t.nn.BatchNorm2d(num_features=out_channels2),
                                             t.nn.ReLU(inplace=True))
        }

        return t.nn.ModuleDict(feature_extractor_modules)

    @staticmethod
    def _define_SSSR_decoder(in_channels1:int, in_channels2:int, mid_channels:int, out_channels:int):
        decoder_modules = \
        {
            'cat_conv': t.nn.Sequential(t.nn.Conv2d(in_channels=(in_channels1+in_channels2),
                                                    out_channels=mid_channels,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=False),
                                        t.nn.BatchNorm2d(num_features=mid_channels),
                                        t.nn.ReLU(inplace=True),
                                        t.nn.Dropout(p=0.5),
                                        t.nn.Conv2d(in_channels=mid_channels,
                                                    out_channels=mid_channels,
                                                    kernel_size=3,
                                                    padding=1,
                                                    bias=False),
                                        t.nn.BatchNorm2d(num_features=mid_channels),
                                        t.nn.ReLU(inplace=True),
                                        t.nn.Dropout(p=0.2)),
            'cls_conv': t.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True),
            # NOTE: Replaced this 'upsample4': t.nn.UpsamplingBilinear2d(scale_factor=4),
            # NOTE: Each 'ConvTranspose2d' scales 2x, so the following modules together scale by 8 times.
            'upsample16_pred': t.nn.Sequential(t.nn.UpsamplingBilinear2d(scale_factor=2.0),     # NOTE: To reduce parameters, we use upsamling here
                                               t.nn.Dropout(p=0.5),
                                               t.nn.ConvTranspose2d(in_channels=out_channels,
                                                                    out_channels=out_channels,
                                                                    kernel_size=2,
                                                                    stride=2,
                                                                    padding=0,
                                                                    bias=False),
                                               t.nn.BatchNorm2d(num_features=out_channels),
                                               t.nn.ReLU(inplace=True),
                                               t.nn.Dropout(p=0.5),
                                               t.nn.ConvTranspose2d(in_channels=out_channels,
                                                                    out_channels=out_channels,
                                                                    kernel_size=2,
                                                                    stride=2,
                                                                    padding=0,
                                                                    bias=True))
        }

        return t.nn.ModuleDict(decoder_modules)

    @staticmethod
    def _define_SISR_decoder(in_channels:int, out_channels:int, upscale_factor:int):
        assert type(upscale_factor) == int, "BUG CHECK: 'upscale_factor' must be an integer type."

        return t.nn.Sequential(t.nn.Conv2d(in_channels=in_channels,
                                           out_channels=(out_channels * (upscale_factor ** 2)),
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=True),
                               t.nn.PixelShuffle(upscale_factor=upscale_factor))

    @staticmethod
    def _define_feature_transformer(in_channels:int, out_channels:int):
        return t.nn.Sequential(t.nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           stride=8,
                                           padding=0,
                                           bias=False),
                               t.nn.BatchNorm2d(num_features=out_channels),
                               t.nn.ReLU(inplace=True))


    def __init__(self, stage, dataset_settings):
        assert stage in [1, 2, 3], "BUG CHECK: Unsupported stage {0} specified in DSRL.__init__().".format(stage)

        # CAUTION: Don't forget to call super's constructor
        super().__init__()

        # Save parameters to class instance variables
        self.stage = stage

        # Feature extractor
        self.feature_extractor = DSRL._define_feature_extractor(in_channels=2048,
                                                                out_channels1=256,
                                                                out_channels2=48)

        # Semantic Segmentation Super Resolution (SSSR)
        self.SSSR_decoder = DSRL._define_SSSR_decoder(in_channels1=256,
                                                      in_channels2=48,
                                                      mid_channels=256,
                                                      out_channels=dataset_settings.NUM_CLASSES)

        if self.stage > 1:
            # Single Image Super-Resolution (SISR)
            self.SISR_decoder = DSRL._define_SISR_decoder(in_channels=(256+48),
                                                          out_channels=consts.NUM_RGB_CHANNELS,
                                                          upscale_factor=8)   # CAUTION: 'upscale_factor' must be integer type

            if self.stage > 2:
                # Feature transform module for SSSR
                self.SSSR_feature_transformer = DSRL._define_feature_transformer(in_channels=dataset_settings.NUM_CLASSES,
                                                                                 out_channels=1)

                # Feature transform module for SISR
                self.SISR_feature_transformer = DSRL._define_feature_transformer(in_channels=consts.NUM_RGB_CHANNELS,
                                                                                 out_channels=1)


    def initialize_with_pretrained_weights(self, weights_dir, map_location=t.device('cpu')):
        self.feature_extractor['backbone'].initialize_with_pretrained_weights(weights_dir, map_location)


    def forward(self, x:t.Tensor):
        with t.autograd.profiler.record_function(DSRL.forward.__qualname__):
            # Extract features
            backbone_features, lowlevel_features = self.feature_extractor['backbone'](x)    # NOTE: Output size (B, 2048, 32, 64), (B, 256, 128, 256)
            aspp_features = self.feature_extractor['aspp'](backbone_features)               # NOTE: Output size (B, 256, 32, 64)
            aspp_features = self.feature_extractor['upsample_sub'](aspp_features)           # NOTE: Output size (B, 256, 128, 256)
            lowlevel_features = self.feature_extractor['shortcut_conv'](lowlevel_features)  # NOTE: Output size (B, 48, 128, 256)
            cat_features = t.cat([aspp_features, lowlevel_features], dim=1)                 # NOTE: Output size (B, 304, 128, 256)

            # Semantic Segmentation Super Resolution (SSSR) decoder
            SSSR_output = self.SSSR_decoder['cat_conv'](cat_features)                       # NOTE: Output size (B, 256, 128, 256)
            SSSR_output = self.SSSR_decoder['cls_conv'](SSSR_output)                        # NOTE: Output size (B, 20, 128, 256)
            SSSR_output = self.SSSR_decoder['upsample16_pred'](SSSR_output)                 # NOTE: Output size (B, 20, 1024, 2048)

            SISR_output = t.zeros(1, requires_grad=False)
            SSSR_transform_output = t.zeros(1, requires_grad=False)
            SISR_transform_output = t.zeros(1, requires_grad=False)
            if self.stage > 1:
                # Single Image Super-Resolution (SISR) decoder
                SISR_output = self.SISR_decoder(cat_features)                           # NOTE: Output size (B, 3, 1024, 2048)

                if self.stage > 2:
                    # Feature transform module for SSSR
                    SSSR_transform_output = self.SSSR_feature_transformer(SSSR_output)   # NOTE: Output size (B, 1, 256, 128)

                    # Feature transform module for SISR
                    SISR_transform_output = self.SISR_feature_transformer(SISR_output)   # NOTE: Output size (B, 1, 256, 128)

            return SSSR_output, SISR_output, SSSR_transform_output, SISR_transform_output