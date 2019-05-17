"""evaluation metrics"""

import torch
import torch.nn as nn
from math import log10
import model.pytorch_ssim as pytorch_ssim

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