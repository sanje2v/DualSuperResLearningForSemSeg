import torch as t
import torchvision as tv


class DSRLSS(t.nn.Module):
    MODEL_INPUT_SIZE = (512, 1024)
    MODEL_OUTPUT_SIZE = (1024, 2048)

    @staticmethod
    def __define_feature_extractor():
        tv.models.vgg16
        module_list = \
        [t.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
         t.nn.ReLU()]

        return t.nn.Sequential(*module_list)

    @staticmethod
    def __define_SSSR_decoder():
        module_list = \
        [t.nn.Conv2d()]

        return t.nn.ModuleList(module_list)

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

        ## Semantic Segmentation Super Resolution (SSSR)
        #self.SSSR_decoder = DSRLSS.__define_SSSR_decoder()

        #if self.train:
        #    if self.stage == 3:
        #        # Feature transform module for SSSR
        #        self.SSSR_feature_transform = DSRLSS.__define_feature_transformer()

        #    if self.stage in [2, 3]:
        #        # Single Image Super-Resolution (SISR)
        #        self.SISR_decoder = DSRLSS.__define_SISR_decoder()

        #        if self.stage == 3:
        #            # Feature transform module for SISR
        #            self.SISR_feature_transform = DSRLSS.__define_feature_transformer()

    def forward(self, x):
        # Extract features
        for layer in self.feature_extractor:
            x = layer(x)
        features = x

        # Semantic Segmentation Super Resolution (SSSR)
        SSSR_output = self.SSSR_decoder(features)

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
        else:
            SISR_output = None

        return SSSR_output, SISR_output
