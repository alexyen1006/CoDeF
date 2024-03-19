from torch import nn
import torch
import torchvision
from einops import rearrange, reduce, repeat


class MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, sigmas, targets):        
        loss = 0.5 * ((inputs - targets) ** 2).mean(-1) * torch.exp(-sigmas) + 0.5 * sigmas
        loss = loss.mean()
        return self.coef * loss

def uncer_consis(sigmas,mk_t):
    mask = mk_t == 1
    
    # Apply slicing to ensure compatible shapes for element-wise operations
    sigmas_slice = sigmas[1:-1, 1:-1]  # Use the appropriate slicing based on your reshaped arrays
    
    # Compute absolute differences with neighboring pixels
    var = (np.abs(sigmas_slice - sigmas[:-2, 1:-1]) +
           np.abs(sigmas_slice - sigmas[2:, 1:-1]) +
           np.abs(sigmas_slice - sigmas[1:-1, :-2]) +
           np.abs(sigmas_slice - sigmas[1:-1, 2:]))
    
    # Apply mask to select relevant pixels
    var = var[mask[1:-1, 1:-1]]
    
    # Compute loss and count of relevant pixels
    loss = np.sum(var) / 4
    cnt = np.sum(mask)
    
    # Normalize the loss
    loss /= (cnt + 1e-8)
    
    return loss


def rgb_to_gray(image):
    gray_image = (0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] +
                  0.114 * image[:, 2, :, :])
    gray_image = gray_image.unsqueeze(1)

    return gray_image


def compute_gradient_loss(pred, gt, mask):
    assert pred.shape == gt.shape, "a and b must have the same shape"

    pred = rgb_to_gray(pred)
    gt = rgb_to_gray(gt)

    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)

    gradient_a_x = torch.nn.functional.conv2d(pred.repeat(1,3,1,1), sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    gradient_a_y = torch.nn.functional.conv2d(pred.repeat(1,3,1,1), sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    # gradient_a_magnitude = torch.sqrt(gradient_a_x ** 2 + gradient_a_y ** 2)

    gradient_b_x = torch.nn.functional.conv2d(gt.repeat(1,3,1,1), sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    gradient_b_y = torch.nn.functional.conv2d(gt.repeat(1,3,1,1), sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
    # gradient_b_magnitude = torch.sqrt(gradient_b_x ** 2 + gradient_b_y ** 2)

    pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
    gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)

    gradient_difference = torch.abs(pred_grad - gt_grad).mean(dim=1,keepdim=True)[mask].sum()/(mask.sum()+1e-8)

    return gradient_difference


loss_dict = {'mse': MSELoss}

import cv2
import numpy as np
def visualize_uncertainty_numpy(sigma, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    sigma: (H, W)
    """
    x = np.nan_to_num(sigma) # change nan to 0
    if minmax is None:
        mi = np.min(x) # get minimum positive sigma (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]
