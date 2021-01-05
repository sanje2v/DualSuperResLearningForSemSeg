import torch as t


class ASPP(t.nn.Module):
    def __init__(self, in_channels, out_channels, rate=1):
        super().__init__()

        branch_params = \
        [
            {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 1, 'padding': 0*rate, 'dilation': 1*rate},
            {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 3, 'padding': 6*rate, 'dilation': 6*rate},
            {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 3, 'padding': 12*rate, 'dilation': 12*rate},
            {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 3, 'padding': 18*rate, 'dilation': 18*rate},
            {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': 1, 'padding': 0, 'dilation': 1},
            {'in_channels': 5*out_channels, 'out_channels': out_channels, 'kernel_size': 1, 'padding': 0, 'dilation': 1}
        ]
        self.branches = t.nn.ModuleList()
        for branch_param in branch_params:
            self.branches.append(t.nn.Sequential(t.nn.Conv2d(**branch_param, bias=False),
                                                 t.nn.BatchNorm2d(num_features=out_channels),
                                                 t.nn.ReLU(inplace=True),
                                                 t.nn.Dropout(p=0.2)))
        self.avg = t.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        branch_outputs = [self.branches[i](x) for i in range(4)]

        global_feature = self.avg(x)
        global_feature = self.branches[4](global_feature)
        global_feature = t.nn.functional.interpolate(global_feature, size=x.size()[-2:], mode='bilinear', align_corners=True)
        branch_outputs.append(global_feature)

        return self.branches[5](t.cat(branch_outputs, dim=1))