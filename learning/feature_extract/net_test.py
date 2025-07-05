import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# from torch.nn import densenet # If you were using densenet, this is how you'd import it

def create_network(settings):
    net = PlaNet(settings)
    return net

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def create(self):
        self._create()

    def forward(self, x):
        return self._internal_call(x)

    def _create(self):
        raise NotImplementedError

    def _internal_call(self):
        raise NotImplementedError


class PlaNet(Network):
    def __init__(self, config):
        super(PlaNet, self).__init__()
        self.config = config
        self._create(input_size=(self.config.img_height,
                                 self.config.img_width,
                                 3 * self.config.use_rgb + 3 * self.config.use_depth))

    def _create(self, input_size, has_bias=True):
        """Init.
        Args:
            input_size (tuple): size of input (H, W, C)
            has_bias (bool, optional): Defaults to True. Conv1d bias?
        """
        if self.config.use_rgb or self.config.use_depth:
            # PyTorch MobileNet expects 3 input channels, so we'll adapt the input layer
            self.backbone = mobilenet_v2(weights='IMAGENET1K_V1')
            # Modify the first convolution layer if the input channels are not 3
            in_channels_backbone = input_size[2]
            if in_channels_backbone != 3:
                first_conv_layer = self.backbone.features[0][0]
                new_first_conv = nn.Conv2d(in_channels_backbone,
                                           first_conv_layer.out_channels,
                                           kernel_size=first_conv_layer.kernel_size,
                                           stride=first_conv_layer.stride,
                                           padding=first_conv_layer.padding,
                                           bias=first_conv_layer.bias is not None)
                self.backbone.features[0][0] = new_first_conv

            # Remove the classifier (top) part
            self.backbone.classifier = nn.Identity()

            if self.config.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # Calculate the output features from the backbone
            # A dummy tensor to determine the output shape
            dummy_input = torch.randn(1, in_channels_backbone, input_size[0], input_size[1])
            with torch.no_grad():
                backbone_output_channels = self.backbone.features(dummy_input).shape[1]

            # reduce a bit the size
            self.resize_op = nn.Sequential(
                nn.Conv1d(backbone_output_channels, 128, kernel_size=1, stride=1, padding='valid', bias=has_bias)
            )

            f = 1.0
            self.img_mergenet = nn.Sequential(
                nn.Conv1d(int(128 * f), int(128 * f), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(128 * f), int(64 * f), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(64 * f), int(64 * f), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(64 * f), int(32 * f), kernel_size=2, stride=1, padding='same', dilation=1),
                nn.LeakyReLU(negative_slope=1e-2)
            )

            self.resize_op_2 = nn.Sequential(
                nn.Conv1d(32, self.config.modes, kernel_size=3, stride=1, padding='valid', bias=has_bias)
            )

        g = 1.0
        self.states_conv = nn.Sequential(
            nn.Conv1d(self.config.imu_dim, int(64 * g), kernel_size=2, stride=1, padding='same', dilation=1),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(64 * g), int(32 * g), kernel_size=2, stride=1, padding='same', dilation=1),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(32 * g), int(32 * g), kernel_size=2, stride=1, padding='same', dilation=1),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(32 * g), int(32 * g), kernel_size=2, stride=1, padding='same', dilation=1)
        )

        self.resize_op_3 = nn.Sequential(
            nn.Conv1d(32, self.config.modes, kernel_size=3, stride=1, padding='valid', bias=has_bias)
        )

        # State dim = 3 (x,y,z) +  alpha
        if len(self.config.predict_state_number) == 0:
            out_len = self.config.out_seq_len
        else:
            out_len = 1
        output_dim = self.config.state_dim * out_len + 1

        g = 1.0
        self.plan_module = nn.Sequential(
            nn.Conv1d(self.config.modes * (32 + (32 if (self.config.use_rgb or self.config.use_depth) else 0)), int(64 * g), kernel_size=1, stride=1, padding='valid'),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(64 * g), int(128 * g), kernel_size=1, stride=1, padding='valid'),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(128 * g), int(128 * g), kernel_size=1, stride=1, padding='valid'),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(128 * g), output_dim, kernel_size=1, stride=1, padding='same')
        )


    def _conv_branch(self, image):
        # image: (batch_size, H, W, C)
        # Permute to (batch_size, C, H, W) for PyTorch Conv2d
        x = self._pf(image)
        x = x.permute(0, 3, 1, 2)
        x = self.backbone.features(x)
        x = x.view(x.shape[0], x.shape[1], -1) # (batch_size, C, H*W)
        x = x.permute(0, 2, 1) # (batch_size, H*W, C)
        x = self.resize_op(x.permute(0, 2, 1)).permute(0, 2, 1) # (batch_size, H*W, 128)
        # x [batch_size, H*W, 128]
        x = x.reshape(x.shape[0], -1)  # (batch_size, H*W*128)
        return x

    def _image_branch(self, img_seq):
        # img_seq: (seq_len, batch_size, img_height, img_width, N)
        seq_len, batch_size, H, W, N = img_seq.shape
        # Reshape to process each frame independently through _conv_branch
        img_seq_reshaped = img_seq.view(seq_len * batch_size, H, W, N)
        img_fts = self._conv_branch(img_seq_reshaped) # (seq_len * batch_size, MxMxC)
        img_fts = img_fts.view(seq_len, batch_size, -1) # (seq_len, batch_size, MxMxC)

        # img_fts (seq_len, batch_size, MxMxC)
        img_fts = img_fts.permute(1, 0, 2)  # batch_size, seq_len, MxMxC
        x = img_fts
        x = x.permute(0, 2, 1) # batch_size, MxMxC, seq_len
        for f in self.img_mergenet:
            x = f(x)
        # final x (batch_size, 32, seq_len)
        x = self.resize_op_2(x)
        # final x (batch_size, modes, 32)
        x = x.permute(0, 2, 1) # (batch_size, 32, modes) -> (batch_size, modes, 32) after permute
        return x

    def _imu_branch(self, embeddings):
        # embeddings: [B, seq_len, D]
        x = embeddings.permute(0, 2, 1) # [B, D, seq_len]
        for f in self.states_conv:
            x = f(x)
        # x (batch_size, 32, seq_len)
        x = self.resize_op_3(x)
        # final x # [batch_size, modes, 32]
        x = x.permute(0, 2, 1) # (batch_size, modes, 32)
        return x

    def _plan_branch(self, embeddings):
        # embeddings: [B, modes, D_total]
        x = embeddings.permute(0, 2, 1) # [B, D_total, modes]
        for f in self.plan_module:
            x = f(x)
        # x: [B, output_dim, modes]
        return x.permute(0, 2, 1) # [B, modes, output_dim]

    def _pf(self, images):
        # PyTorch MobileNet pre-processing is typically handled by transforms
        # For direct input, assume values are in [0, 255] and normalize to [-1, 1] as MobileNet expects.
        # https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
        # All pre-trained models expect input images normalized in the same way,
        # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        # The images have to be loaded in to a range of [0, 1] and then normalized
        # using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        # Since the original TF code uses tf.keras.applications.mobilenet.preprocess_input
        # which scales inputs to [-1, 1], we will replicate that here.
        return (images / 127.5) - 1.0


    def _preprocess_frames(self, inputs):
        img_seq = None
        if self.config.use_rgb and self.config.use_depth:
            # Inputs are (batch_size, seq_len, img_height, img_width, channels)
            # Need to concatenate along the last dimension (channels)
            img_seq = torch.cat((inputs['rgb'], inputs['depth']),
                                dim=-1)  # (batch_size, seq_len, img_height, img_width, 6)
        elif self.config.use_rgb and (not self.config.use_depth):
            img_seq = inputs['rgb']  # (batch_size, seq_len, img_height, img_width, 3)
        elif self.config.use_depth and (not self.config.use_rgb):
            # For depth-only, assuming depth is 3 channels, if not, adjust input_size
            img_seq = inputs['depth']  # (batch_size, seq_len, img_height, img_width, 3)
        else:
            return None
        
        # One of them passed, so need to process it
        # Transpose to (seq_len, batch_size, img_height, img_width, N)
        img_seq = img_seq.permute(1, 0, 2, 3, 4)
        img_embeddings = self._image_branch(img_seq)
        return img_embeddings

    def _internal_call(self, inputs):
        # Determine the IMU observation dimension based on config
        if self.config.use_position:
            imu_obs = inputs['imu'] # (batch_size, seq_len, total_imu_dim)
            self.config.imu_dim = imu_obs.shape[-1]
        else:
            # always pass z
            imu_obs = inputs['imu'][:, :, 3:]
            self.config.imu_dim = imu_obs.shape[-1]

        if not self.config.use_attitude:
            if self.config.use_position:
                print("ERROR: Do not use position without attitude!")
                return
            else:
                imu_obs = inputs['imu'][:, :, 12:] # velocity and optionally body rates
                self.config.imu_dim = imu_obs.shape[-1]

        imu_embeddings = self._imu_branch(imu_obs) # [B, modes, 32]
        img_embeddings = self._preprocess_frames(inputs) # [B, modes, 32]

        if img_embeddings is not None:
            total_embeddings = torch.cat((img_embeddings, imu_embeddings), dim=-1)  # [B, modes, 32 + 32]
        else:
            total_embeddings = imu_embeddings # [B, modes, 32]

        output = self._plan_branch(total_embeddings) # [B, modes, output_dim]
        return output