import torch
import torch.nn as nn

from models.model_base import InitializationMixin


class UNet(nn.Module, InitializationMixin):
    """
    UNet from arXiv:1505.04597 U-Net: Convolutional Networks for Biomedical Image
    Segmentation
    Attention block from arXiv:1804.03999 Attention U-Net: Learning Where to Look for the
    Pancreas

    Parameters
    ----------
    classes: int
        Number of classes to predict for.
    img_channels: int
        Number of input image channels.
    model_depth: str
        Name for the number of blocks in the encoder.
    encoder_arch_config: dict
        Further configurations for blocks in the encoder.
    unet_arch_config: dict
        Further configurations for blocks in unet.
        Default = {downsample: maxpool, upsample: transpose, unet_block: unetblock,
                norm: nn.BatchNorm2d, act: nn.ReLU(inplace=True), attention: False}
    pretrained_encoder: bool or str
        True or path pretrains the encoder accordingly.
    pretrained_unet: str
        Path loads the state_dict of the encoder, bridge, and decoder from a file.

    Note
    ----
    Encoder requires an indexable block for each resolution (reuires 4 total).
    Each final block must have .out_channels: int as an attribute
    """
    def __init__(self, classes, img_channels,
            model_depth=None, encoder_arch_config=None, unet_arch_config=None,
            pretrained_encoder=False, pretrained_unet=False):
        super().__init__()
        self.classes = classes
        self.img_channels = img_channels
        # encoder settings
        self.model_depth = model_depth.lower() if isinstance(model_depth, str) \
                else model_depth
        self.encoder_arch_config = {} if encoder_arch_config is None \
                else encoder_arch_config
        self.pretrained_encoder = pretrained_encoder
        # unet settings
        default_unet_arch_config = {
                'downsample': 'maxpool',
                'upsample': 'transpose',
                'block': 'unetblock',
                'norm': nn.BatchNorm2d,
                'act': nn.ReLU(inplace=True),
                'attention': False,
        }
        unet_arch_config = {} if unet_arch_config is None else unet_arch_config
        self.unet_arch_config = default_unet_arch_config | unet_arch_config

        self.assemble()
        if not pretrained_encoder:
            self.initialize_parameters(self.encoder)
        self.initialize_parameters(self.bridge)
        self.initialize_parameters(self.decoder)
        self.pretrain_file(pretrained_unet)

    def forward(self, x1_1, x1_2):
        # stream 1
        x1_1 = self.encoder[0](x1_1)
        x2_1 = self.encoder[1](x1_1)
        x3_1 = self.encoder[2](x2_1)
        x4_1 = self.encoder[3](x3_1)

        x5_1 = self.bridge(x4_1)

        # stream 2
        x1_2 = self.encoder[0](x1_2)
        x2_2 = self.encoder[1](x1_2)
        x3_2 = self.encoder[2](x2_2)
        x4_2 = self.encoder[3](x3_2)

        x5_2 = self.bridge(x4_2)

        # concat outputs from encoder
        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x3_1 = torch.cat((x3_1, x3_2), dim=1)
        x4_1 = torch.cat((x4_1, x4_2), dim=1)
        x5_1 = torch.cat((x5_1, x5_2), dim=1)

        out = self.decoder[0](x5_1, x4_1)
        out = self.decoder[1](out, x3_1)
        out = self.decoder[2](out, x2_1)
        out = self.decoder[3](out, x1_1)
        out = self.decoder[4](out)
        return out

    def assemble(self):
        self.__assemble_encoder()
        self.__assemble_bridge()
        self.__assemble_decoder()

    def __assemble_encoder(self):
        encoder_list = []
        encoder_list.append(nn.Sequential(
            StemBlock(self.img_channels, **self.unet_arch_config)))
        self.model_depth = [1, 1, 1]
        for b, block_group, in enumerate(self.model_depth):
            for block in range(block_group):
                in_channels = 64 * 2 ** b
                out_channels = in_channels * 2
                encoder_list.append(nn.Sequential(
                    DownBlock(in_channels, out_channels, **self.unet_arch_config))
                )
        self.model_depth.append(1)  # consistency with number of layers
        self.encoder = nn.Sequential(*encoder_list)

    def __assemble_bridge(self):
        # channels are copied from self.encoder
        in_channels = self.encoder[-1][-1].out_channels
        out_channels = in_channels * 2
        self.bridge = DownBlock(in_channels, out_channels, **self.unet_arch_config)

    def __assemble_decoder(self):
        # channels are copied from self.encoder
        decoder_list = []
        for l in range(len(self.encoder)):
            skip_channels = self.encoder[::-1][l][-1].out_channels
            out_channels = skip_channels if l == len(self.encoder) - 1 \
                    else self.encoder[::-1][l][0].in_channels * 2
            decoder_list.append(
                    UpBlock(skip_channels, out_channels, **self.unet_arch_config)
            )
        decoder_list.append(LastConvBlock(out_channels * 2, self.classes))
        self.decoder = nn.Sequential(*decoder_list)


def set_block(block, *args):
    block_dict = {'unetblock': UNetBlock, 'unetblock_reverse': UNetBlockReverse}
    return block_dict[block](*args)


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64,
            block='unetblock', norm=nn.BatchNorm2d, norm_args={},
            act=nn.ReLU(inplace=True), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem = self.__configure(block, norm, norm_args, act)

    def forward(self, x):
        return self.stem(x)

    def __configure(self, block, norm, norm_args, act):
        return set_block(block, self.in_channels, self.out_channels, 1, norm, norm_args,
                act)


class UNetBlock(nn.Module):
    #  conv -> bn -> relu -> conv -> bn -> relu
    def __init__(self, in_channels, out_channels, stride=1,
            norm=nn.BatchNorm2d, norm_args={}, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.act = act
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            norm(out_channels, **norm_args),
            self.act,
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            norm(out_channels, **norm_args),
            self.act,
        )

    def forward(self, x):
        return self.block(x)


class UNetBlockReverse(nn.Module):
    #  conv -> relu -> bn -> conv -> relu -> bn
    def __init__(self, in_channels, out_channels, stride=1,
            norm=nn.BatchNorm2d, norm_args={}, act=nn.ReLU(inplace=True), **kwargs):
        super().__init__()
        self.act = act
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            self.act,
            norm(out_channels, **norm_args),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            self.act,
            norm(out_channels, **norm_args),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    #  maxpool -> UNetBlock
    def __init__(self, in_channels, out_channels, downsample='maxpool',
            block='unetblock', norm=nn.BatchNorm2d, norm_args={},
            act=nn.ReLU(inplace=True), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down, self.block = self.__configure(downsample, block, norm, norm_args, act)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)

    def __configure(self, downsample, block, norm, norm_args, act):
        downsample_configuration_dict = {
                'maxpool': nn.MaxPool2d(2),
                'avgpool': nn.AvgPool2d(2),
        }
        if downsample in downsample_configuration_dict:
            down = downsample_configuration_dict[downsample]
            stride = 1
        elif downsample == 'stride':
            down = nn.Sequential()
            stride = 2
        return down, set_block(
                block, self.in_channels, self.out_channels, stride, norm, norm_args, act)


class UpBlock(nn.Module):
    #  upsample -> UNetBlock
    def __init__(self, skip_channels, out_channels, upsample='transpose',
            block='unetblock', norm=nn.BatchNorm2d, norm_args={},
            act=nn.ReLU(inplace=True), attention=False, **kwargs):
        super().__init__()
        self.skip_channels = skip_channels * 2
        self.out_channels = out_channels * 2
        self.attention = attention
        if self.attention:
            self.attention_block = AttentionBlock(
                    self.skip_channels,
                    self.skip_channels,
                    norm,
                    norm_args,
                    act,
            )
        self.up, self.block = self.__configure(upsample, block, norm, norm_args, act)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)
        if self.attention:
            x_up = self.attention_block(x_up, x_skip)  # attends up
        x_up = torch.cat([x_up, x_skip], dim=1)
        return self.block(x_up)

    def __configure(self, upsample, block, norm, norm_args, act):
        upsample_configuration_dict = {
                'bilinear': nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(self.skip_channels * 2, self.skip_channels, 1),
                ),
                'transpose': nn.ConvTranspose2d(self.skip_channels * 2,
                    self.skip_channels, 2, 2),
                'subpixel': nn.Sequential(
                    nn.Conv2d(self.skip_channels * 2, self.skip_channels * 4, 1),
                    nn.PixelShuffle(2),
                ),
        }
        upsample_block = upsample_configuration_dict[upsample]
        conv_block = set_block(
                block, self.skip_channels * 2, self.out_channels, 1, norm, norm_args, act)
        return upsample_block, conv_block


class LastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.last_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.last_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm2d, norm_args={},
            act=nn.ReLU(inplace=True)):
        super().__init__()
        self.act = act
        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            norm(out_channels, **norm_args),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            norm(out_channels, **norm_args),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, bias=True),
            norm(1, **norm_args),
            nn.Sigmoid(),
        )

    def forward(self, x_up, x_skip):
        x_skip = self.conv_up(x_up) + self.conv_skip(x_skip)
        x_skip = self.act(x_up)
        x_skip = self.attention(x_up)
        return x_up * x_skip


if __name__ == '__main__':
    torch.manual_seed(0)
    print('--- Testing random array ---')
    model = UNet(2, 3, None, None, {'uspample': 'bilinear'}).eval()
    print(model)


