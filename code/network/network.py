import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Implements channel attention mechanism as part of CBAM, focusing on inter-channel relationship.
    It calculates attention weights based on the channel-wise global average and max pooling,
    enabling the network to focus more on informative features.

    Args:
        num_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio to control the bottleneck size in the fully connected layers.
    """

    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1) * x


class ChannelPool(nn.Module):
    """
    Performs channel-wise pooling by taking the maximum and average along the channel dimension,
    creating a tensor with double the number of channels. This tensor is used in spatial attention to
    provide a summarized feature map across channels.
    """

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class BasicConv(nn.Module):
    """
    Basic convolutional block that includes a convolutional layer optionally followed by
    batch normalization and a ReLU activation function.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        relu (bool): If True, applies a ReLU activation after the convolution.
        bn (bool): If True, applies batch normalization after the convolution.
        bias (bool): If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    """
    Implements spatial attention mechanism as part of CBAM, focusing on where to emphasize or suppress
    features spatially. It uses a compressed feature map obtained by channel pooling and applies a convolution
    to produce a spatial attention map.

    Args:
        kernel_size (int): Size of the convolution kernel used for spatial attention.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    """
    CBAM module that sequentially applies channel and spatial attention mechanisms to an input feature map.
    It enhances the representational power of the network by focusing on important features both channel-wise
    and spatially.

    Args:
        num_channels (int): Number of channels in the input feature map.
        reduction_ratio (int): Reduction ratio for the channel attention's fully connected layer.
        attention_kernel_size (int): Kernel size for the spatial attention convolution.
    """

    def __init__(self, num_channels, reduction_ratio=16, attention_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ConvCBAMBlock(nn.Module):
    """
    Convolutional block equipped with CBAM, performing convolution followed by batch normalization,
    ReLU activation, and CBAM attention. It serves as a fundamental building block for constructing
    deeper architectures with attention mechanisms.

    Args:
        in_channels (int): Number of input channels to the convolutional layer.
        out_channels (int): Number of output channels from the convolutional layer.
        reduction_ratio (int): Reduction ratio used in the CBAM channel attention.
        attention_kernel_size (int): Kernel size for the CBAM spatial attention.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        reduction_ratio=16,
        attention_kernel_size=7,
    ):
        super(ConvCBAMBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels, reduction_ratio, attention_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        return x


class Network(nn.Module):
    """
    Defines a neural network with multiple ConvCBAM blocks and an embedding layer for input processing.
    This architecture is suitable for tasks that benefit from attention mechanisms, like visual recognition
    or in cases like Minesweeper for feature enhancement based on contextual importance.

    Args:
        num_embeddings (int): Number of unique embeddings, e.g., different states in Minesweeper.
        embedding_dim (int): Dimensionality of each embedding.
        cbam_channels (int): Number of channels in the CBAM blocks.
    """

    def __init__(self, num_embeddings=11, embedding_dim=4, cbam_channels=128):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.network = nn.Sequential(
            ConvCBAMBlock(embedding_dim, cbam_channels),
            ConvCBAMBlock(cbam_channels, cbam_channels),
            ConvCBAMBlock(cbam_channels, cbam_channels),
            ConvCBAMBlock(cbam_channels, cbam_channels),
            nn.BatchNorm2d(cbam_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=cbam_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.squeeze(1).permute(0, 3, 1, 2)
        x = self.network(x)
        return x
