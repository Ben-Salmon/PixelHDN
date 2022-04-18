import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from PixelCNN.GMM import GMM
import torch.nn.functional as F

class MaskedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, mask, **kwargs):
        """Implements a convolution with mask applied on its weights.

        Args:
            c_in: Number of input channels
            c_out: Number of output channels
            mask: Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs: Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple(dilation * (kernel_size[i] - 1) // 2 for i in range(2))
        # Actual convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
        return self.conv(x)
        
class VerticalStackConvolution(MaskedConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(in_channels, out_channels, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(in_channels, out_channels, mask, **kwargs)

class GatedConv(nn.Module):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.v_conv = VerticalStackConvolution(num_filters, 2*num_filters, kernel_size, **kwargs)
        self.v_conv_1x1 = nn.Conv2d(2*num_filters, 2*num_filters, 1)
        self.h_conv = HorizontalStackConvolution(num_filters, 2*num_filters, kernel_size, **kwargs)
        self.h_conv_1x1 = nn.Conv2d(num_filters, num_filters, 1)
        self.v_s_conv = nn.Conv2d(num_filters, 2*num_filters, 1)
        self.h_s_conv = nn.Conv2d(num_filters, 2*num_filters, 1)

    def forward(self, x_v, x_h, s):
        # Signal convolutions
        s_v = self.v_s_conv(s)
        s_h = self.h_s_conv(s)
        s_v_tan, s_v_sig = s_v.chunk(2, dim=1)
        s_h_tan, s_h_sig = s_h.chunk(2, dim=1)
        
        # Vertical stack
        v_stack_feat = self.v_conv(x_v)
        v_tan, v_sig = v_stack_feat.chunk(2, dim=1)
        v_tan, v_sig = v_tan + s_v_tan, v_sig + s_v_sig
        v_stack_out = torch.tanh(v_tan) * torch.sigmoid(v_sig)
        
        # Horizontal stack
        h_stack_feat = self.h_conv(x_h)
        h_stack_feat = h_stack_feat + self.v_conv_1x1(v_stack_feat)
        h_tan, h_sig = h_stack_feat.chunk(2, dim=1)
        h_tan, h_sig = h_tan + s_h_tan, h_sig + s_h_sig
        h_stack_out = torch.tanh(h_tan) * torch.sigmoid(h_sig)
        h_stack_out = self.h_conv_1x1(h_stack_out)
        h_stack_out = h_stack_out + x_h

        return v_stack_out, h_stack_out

class PixelCNN(GMM):
    """ A conv-net that shifts it's output to modify the receptive field of ouput pixels

    """
    
    def __init__(self, 
                 colour_channels=1, 
                 depth=5,
                 num_filters=128,
                 kernel_size=7,
                 num_gaussians=10,
                 mean=0, std=1):
        """
        Arguments:
            colour_channels: int, number of channels in the input tensor.
            depth: int>=2, number of convolutional layers.
            num_filters: int, number of convolutional filters.
            kernel_size: int, side length of kernel, should be 3,7,11,15,...
            num_gaussians: int, number of components in GMM.
            mean: float, mean of entire noisy dataset.
            std: positive float, std of entire noisy dataset.
        """
        self.save_hyperparameters()
        super().__init__(mean, std, num_gaussians)
        self.depth = depth
        out_channels = colour_channels * num_gaussians * 3
        
        self.v_inconv = VerticalStackConvolution(colour_channels, num_filters, kernel_size, mask_center=True)
        self.h_inconv = HorizontalStackConvolution(colour_channels, num_filters, kernel_size, mask_center=True)
        self.s_inconv = nn.Conv2d(colour_channels, num_filters, 1)
        self.gatedconvs = nn.ModuleList([GatedConv(num_filters, kernel_size)] * depth)
        self.outconv = nn.Conv2d(num_filters, out_channels, 1)
        
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)                     
            
    def forward(self, x, s):
        x_v = self.v_inconv(x)
        x_h = self.h_inconv(x)
        s = self.s_inconv(s)
        for layer in self.gatedconvs:
            x_v, x_h = layer(x_v, x_h, s)
        out = self.outconv(F.elu(x_h))
        
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]