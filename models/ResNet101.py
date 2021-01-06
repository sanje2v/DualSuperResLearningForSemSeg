import torch as t
import torchvision as tv


# NOTE: This code is derivative of 'ResNet' class of PyTorch implemention at 'torchvision.models.resnet'
class ResNet101(t.nn.Module):
    PRETRAINED_WEIGHTS_URL = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"

    def __init__(self, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet101, self).__init__()

        block = tv.models.resnet.Bottleneck # CAUTION: Export hidden by module's '__ALL__' but still is importable
        layers = [3, 4, 23, 3]

        if norm_layer is None:
            norm_layer = t.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation) == 3, "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation)

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = t.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = t.nn.ReLU(inplace=True)
        self.maxpool = t.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, t.nn.Conv2d):
                t.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (t.nn.BatchNorm2d, t.nn.GroupNorm)):
                t.nn.init.constant_(m.weight, 1)
                t.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, tv.models.Bottleneck):
                    t.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, tv.models.BasicBlock):
                    t.nn.init.constant_(m.bn2.weight, 0)

    def initialize_with_pretrained_weights(self, weights_dir):
        pretrained_state_dict = t.utils.model_zoo.load_url(self.PRETRAINED_WEIGHTS_URL,
                                                           weights_dir,
                                                           progress=True,
                                                           file_name='resnet101_pretrained.pth')
        missing_keys, _ = self.load_state_dict(pretrained_state_dict, strict=False)
        assert len(missing_keys) == 0, "BUG CHECK: Pretrained weights from model zoo for ResNet101 has missing keys: {}.".format(missing_keys)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = t.nn.Sequential(
                tv.models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return t.nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features

    def forward(self, x):
        return self._forward_impl(x)