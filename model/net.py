"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log10

import model.pytorch_ssim as pytorch_ssim


class x2Net(nn.Module):
    """
    A dilate network for SR task that expand feature maps first then shrink it to original size.
    """
    def __init__(self, in_channel=1):
        super(x2Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, 3, padding=1)

        self.res1 = self.__res_block(64, 64, 16)
        self.res2 = self.__res_block(48, 48, 16)
        self.res3 = self.__res_block(32, 32, 16)

        self.expand1 = nn.ConvTranspose2d(64, 48, 2, stride=2)  # 128 -> 256  
        self.expand2 = nn.ConvTranspose2d(48, 32, 2, stride=2)  # 256 -> 512
        self.expand3 = nn.ConvTranspose2d(32, 32, 2, stride=2)   # 512 -> 1024

        self.dres1 = self.__res_block(32, 32, 16)                            
        self.dres2 = self.__res_block(32, 32, 16)                            
        
        self.shrink1 = nn.Sequential(
            # nn.Conv2d(64, 32, 3, stride=1, padding=1),             # 1024 -> 512
            nn.AvgPool2d(2, 2)
        )
        self.shrink2 = nn.Sequential(
            # nn.Conv2d(32, 16, 3, stride=1, padding=1),             # 512 -> 256
            nn.AvgPool2d(2, 2)
        )

        self.out_conv = nn.Conv2d(32, 1, 1)
        
    def __res_block(self, in_channel, out_channel, num_filters, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channel, num_filters, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(num_filters, out_channel, 1)
        )

    def forward(self, X):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            X: (Tensor) contains a batch of images, of dimension (batch_size, in_channel, in_height, in_width) .

        Returns:
            out: (Tensor) the predicted high-resolution image tensor.
        """
        # extract Y band from YCbCr composition
        x = torch.unsqueeze(X[:, 0, :, :], 1)
        Cb = torch.unsqueeze(X[:, 1, :, :], 1)
        Cr = torch.unsqueeze(X[:, 2, :, :], 1)
        # save a identity branch
        identity = x
        identity = F.interpolate(identity, scale_factor=2, mode='nearest')

        # first conv to increase depth
        x = F.leaky_relu(self.conv1(x))

        # expand path
        x1_res = self.res1(x)
        x1 = F.leaky_relu(torch.add(x, x1_res))
        x1 = F.leaky_relu(self.expand1(x1))

        x2_res = self.res2(x1)
        x2 = F.leaky_relu(torch.add(x1, x2_res))
        x2 = F.leaky_relu(self.expand2(x2))

        x3_res = self.res3(x2)
        x3 = F.leaky_relu(torch.add(x2, x3_res))
        x3 = F.leaky_relu(self.expand3(x3))

        # shrink path
        x4_res = self.dres1(x3)
        x4 = F.leaky_relu(torch.add(x3, x4_res))
        x4 = F.leaky_relu(self.shrink1(x4))

        x5_res = self.dres2(x4)
        x5 = F.leaky_relu(torch.add(x4, x5_res))
        x5 = F.leaky_relu(self.shrink2(x5))

        # output
        x_out = torch.tanh(self.out_conv(x5))   # use tanh for -1~1 range residual
        x_out = torch.add(identity, x_out)

        # add back Cb Cr band
        Cb = F.interpolate(Cb, scale_factor=2, mode='nearest')
        Cr = F.interpolate(Cr, scale_factor=2, mode='nearest')

        out = torch.cat([x_out, Cb, Cr], dim=1)

        return out

class x4Net(nn.Module):
    """
    A dilate network for SR task that expand feature maps first then shrink it to original size.
    """
    def __init__(self, in_channel=1):
        super(x4Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, 3, padding=1)

        self.res1 = self.__res_block(64, 64, 16)
        self.res2 = self.__res_block(64, 64, 16)
        self.res3 = self.__res_block(32, 32, 16)
        self.res4 = self.__res_block(32, 32, 16)

        self.expand1 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 64 -> 128  
        self.expand2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 128 -> 256
        self.expand3 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # 256 -> 512
        self.expand4 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # 512 -> 1024

        self.dres1 = self.__res_block(32, 32, 16)                            
        self.dres2 = self.__res_block(32, 32, 16)                            
        
        self.shrink1 = nn.Sequential(
            # nn.Conv2d(64, 32, 3, stride=1, padding=1),             # 1024 -> 512
            nn.AvgPool2d(2, 2)
        )
        self.shrink2 = nn.Sequential(
            # nn.Conv2d(32, 16, 3, stride=1, padding=1),             # 512 -> 256
            nn.AvgPool2d(2, 2)
        )

        self.out_conv = nn.Conv2d(32, 1, 1)
        
    def __res_block(self, in_channel, out_channel, num_filters, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channel, num_filters, 1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(num_filters, out_channel, 1)
        )

    def forward(self, X):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            X: (Tensor) contains a batch of images, of dimension (batch_size, in_channel, in_height, in_width) .

        Returns:
            out: (Tensor) the predicted high-resolution image tensor.
        """
        # extract Y band from YCbCr composition
        x = torch.unsqueeze(X[:, 0, :, :], 1)
        Cb = torch.unsqueeze(X[:, 1, :, :], 1)
        Cr = torch.unsqueeze(X[:, 2, :, :], 1)
        # save a identity branch
        identity = x
        identity = F.interpolate(identity, scale_factor=4, mode='nearest')

        # first conv to increase depth
        x = F.leaky_relu(self.conv1(x))

        # expand path
        x1_res = self.res1(x)
        x1 = F.leaky_relu(torch.add(x, x1_res))
        x1 = F.leaky_relu(self.expand1(x1))

        x2_res = self.res2(x1)
        x2 = F.leaky_relu(torch.add(x1, x2_res))
        x2 = F.leaky_relu(self.expand2(x2))

        x3_res = self.res3(x2)
        x3 = F.leaky_relu(torch.add(x2, x3_res))
        x3 = F.leaky_relu(self.expand3(x3))

        # reuse x3 variable name for convenience
        x3_res = self.res4(x3)
        x3 = F.leaky_relu(torch.add(x3, x3_res))
        x3 = F.leaky_relu(self.expand4(x3))

        # shrink path
        x4_res = self.dres1(x3)
        x4 = F.leaky_relu(torch.add(x3, x4_res))
        x4 = F.leaky_relu(self.shrink1(x4))

        x5_res = self.dres2(x4)
        x5 = F.leaky_relu(torch.add(x4, x5_res))
        x5 = F.leaky_relu(self.shrink2(x5))

        # output
        x_out = torch.tanh(self.out_conv(x5))   # use tanh for -1~1 range residual
        x_out = torch.add(identity, x_out)

        # add back Cb Cr band
        Cb = F.interpolate(Cb, scale_factor=4, mode='nearest')
        Cr = F.interpolate(Cr, scale_factor=4, mode='nearest')

        out = torch.cat([x_out, Cb, Cr], dim=1)

        return out


# loss function
def loss_fn(yPred, yTrue):
    """
    Compute the loss given output image and ground truth image.

    Args:
        yPred: (Tensor) model output, size (batch_size, in_channel, in_height*scale, in_width*scale)
        yTrue: (Tensor) ground truth image, same size as outputs

    Returns:
        loss (Tensor): loss score for all images in the batch
    """
    return nn.MSELoss(reduction='mean')(yPred, yTrue)
    
# evaluation metrics
def psnr(yPred, yTrue):
    """
    Compute the PSNR (Peak-Signal-Noise-Ratio) given output image and ground truth image.

    Args:
        yPred: (Tensor) model output, size (batch_size, in_channel, in_height*scale, in_width*scale)
        yTrue: (Tensor) ground truth image, same size as outputs

    Returns:
        psnr (Tensor): loss score for all images in the batch
    """
    with torch.no_grad():
        mse = nn.MSELoss(reduction='mean')(yPred, yTrue)
        psnr = 10 * log10(1/mse.data)

    return psnr

def ssim(yPred, yTrue):
    return pytorch_ssim.ssim(yPred, yTrue)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'psnr': psnr,
    'ssim': ssim
}
