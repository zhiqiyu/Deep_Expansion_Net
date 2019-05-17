"""loss function"""

import torch.nn as nn

def mse(yPred, yTrue):
    """
    Compute the loss given output image and ground truth image.

    Args:
        yPred: (Tensor) model output, size (batch_size, in_channel, in_height*scale, in_width*scale)
        yTrue: (Tensor) ground truth image, same size as outputs

    Returns:
        loss (Tensor): loss score for all images in the batch
    """
    return nn.MSELoss(reduction='mean')(yPred, yTrue)


loss_fn = mse