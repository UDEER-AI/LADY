import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """
    Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(
        loss: Tensor,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        avg_factor: Optional[float] = None
    ) -> Tensor:
    """
    Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
        reduction (str, optional): Same as built-in losses of PyTorch.
        avg_factor (Optional[float], optional): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(
        pred,
        target,
        weight=None,
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        avg_factor=None
    ):
    """
    PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating factor.
        alpha (float, optional): A balanced form for Focal Loss. 
        reduction (str, optional): The method used to reduce the loss into a scalar.
        avg_factor (int, optional): Average factor that is used to average the loss.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)

    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ), which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g. in FSAF. 
                # But it may be flattened of shape (num_priors x num_class, ), while loss is still of shape (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss
