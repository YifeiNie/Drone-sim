import torch
import torch.nn as nn
import torchvision.models
class Network(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create(self):
        self._create()

    def forward(self, x):
        self._forward(x)

    def _create(self):
        raise NotImplementedError

    def _internal_forward(self, x):
        raise NotImplementedError
    

class Net(Network):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._create()

    def _create(self, input_size, has_bias=True):


        # feature extraction backbone for depth img
        self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        if(input_size[2] != 3):
            old_first_conv_layer = self.backbone.features[0][0]
            new_first_conv_layer = nn.Conv2d(input_size[2],
                                        out_channels = old_first_conv_layer.out_channels,
                                        kernel_size  = old_first_conv_layer.kernel_size,
                                        stride       = old_first_conv_layer.stride,
                                        padding      = old_first_conv_layer.padding,
                                        bias         = old_first_conv_layer.bias is not None)
        self.backbone.features[0][0] = new_first_conv_layer
        self.backbone.classifier = nn.Identity()

        if self.config.free_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        dummy_input = torch.randn(1, input_size[2], input_size[0], input_size[1])
        with torch.no_grad():
            backbone_output_channels = self.backbone.features(dummy_input).shape[1]

        # dim reduction for depth img
        self.resize_op = nn.Sequential(
                nn.Conv1d(backbone_output_channels, 128, kernel_size=1, stride=1, padding='valid', bias=has_bias)
        )

        # merge for depth img
        self.img_mergenet = nn.Sequential(
                nn.Conv1d(int(128), int(128), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(128), int(64), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(64), int(64), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(64), int(32), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2)
        )

        self.resize_op_2 = nn.Sequential(
            nn.Conv1d(32, self.config.modes, kernel_size=3, stride=1, padding='valid', bias=has_bias)
        )

        # 模拟输出维度
        output_dim = self.config.state_dim * self.config.out_seq_len + 1
        self.output_layer = nn.Conv1d(32, output_dim, kernel_size=1)
















        