from math import sqrt
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from kornia.geometry import epipolar
from kornia.utils import create_meshgrid


def compute_pseudo_gt_corrs(
    pred_conf: torch.Tensor,
    fundamental_matrix: torch.Tensor,
    epipolar_line_threshold: float = sqrt(2.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the pseudo-ground truth correspondences by choosing the maximum confidence location along the epipolar line.

    Parameters
    ----------
    pred_conf (batch_size x height0 x width0 x height1 x width1 tensor):
        For each pixel (patch) in image 0, this gives a confidence matrix for the probability that the respective pixel (patch) in image 1 is the matching pixel (patch). If any values are outside the range [0, 1], a dual-softmax will be applied (soft nearest-neighbours).

    fundamental_matrix (batch_size x 3 x 3 tensor):
        The fundamental matrix from 0 to 1 for the respective image pair.

    epipolar_line_threshold (float):
        The threshold for the distance of a pixel to the epipolar line to be considered to be part of the epipolar line. By default sqrt(2.0).

    Returns
    -------
    pseudo_gt (batch_size x height0 x width0 x height1 x width1 tensor):
        The pseudo-ground truth correspondences, where for each pixel (patch) in image 0, the pixel (patch) in image 1 with the maximum confidence along the epipolar line is marked with a 1.

    visible_epiline_mask (batch_size x height0 x width0 x 1 x 1 bool tensor):
        A mask indicating whether the epipolar line for each pixel (patch) in image 0 is visible in image 1.
    """

    b, h0, w0, h1, w1 = pred_conf.shape
    dims = {"b": b, "h0": h0, "w0": w0, "h1": h1, "w1": w1}
    device = pred_conf.device

    # compute epipolar line mask using thresholding
    grid0 = repeat(
        create_meshgrid(h0, w0, False, device=device),
        "() h0 w0 two -> b (h0 w0 h1 w1) two",
        **dims,
        two=2,
    )
    grid1 = repeat(
        create_meshgrid(h1, w1, False, device=device),
        "() h1 w1 two -> b (h0 w0 h1 w1) two",
        **dims,
        two=2,
    )

    dist_0epiline_to_1points = epipolar.left_to_right_epipolar_distance(
        grid0, grid1, fundamental_matrix
    )
    dist_0epiline_to_1points = rearrange(
        dist_0epiline_to_1points,
        "b (h0 w0 h1 w1) -> b h0 w0 h1 w1",
        **dims,
    )

    epilines_mask = dist_0epiline_to_1points < epipolar_line_threshold

    # normalise the confidence matrix if necessary by taking the dual softmax
    if (pred_conf < 0).any() or (pred_conf > 1).any():
        pred_conf = rearrange(pred_conf, "b h0 w0 h1 w1 -> b (h0 w0) (h1 w1)", **dims)
        pred_conf = F.softmax(pred_conf, dim=1) * F.softmax(pred_conf, dim=2)
        pred_conf = rearrange(pred_conf, "b (h0 w0) (h1 w1) -> b h0 w0 h1 w1", **dims)

    # compute pseudo-ground truth by choosing the point with the maximum prediction along epipolar line
    temp_conf = pred_conf.detach().clone()
    temp_conf[epilines_mask] = -1
    max_conf_on_epilines = reduce(
        temp_conf, "b h0 w0 h1 w1 -> b h0 w0", reduction="max", **dims
    )
    max_conf_on_epilines = repeat(
        max_conf_on_epilines, "b h0 w0 -> b h0 w0 h1 w1", **dims
    )
    pseudo_gt = (temp_conf == max_conf_on_epilines).float()

    visible_epiline_mask = epilines_mask.any(dim=(-1, -2), keepdim=True)

    return pseudo_gt, visible_epiline_mask


def epipolar_classification_loss(
    pred_conf: torch.Tensor,
    fundamental_matrix: torch.Tensor,
    epipolar_line_threshold: float = sqrt(2.0),
    loss_type: Literal["cross_entropy"] = "cross_entropy",
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Compute the epipolar classification loss for a batch of predicted match confidences and fundamental matrices.

    Parameters
    ----------
    pred_conf (batch_size x height0 x width0 x height1 x width1 tensor):
        For each pixel (patch) in image 0, this gives a confidence matrix for the probability that the respective pixel (patch) in image 1 is the matching pixel (patch). If any values are outside the range [0, 1], a dual-softmax will be applied (soft nearest-neighbours).

    fundamental_matrix (batch_size x 3 x 3 tensor):
        The fundamental matrix from 0 to 1 for the respective image pair.

    epipolar_line_threshold (float):
        The threshold for the distance of a pixel to the epipolar line to be considered to be part of the epipolar line. By default sqrt(2.0).

    loss_type (str):
        Only "cross_entropy" implemented at the moment. Describes the type of classification loss to use. By default "cross_entropy".

    reduction (str):
        One of "mean", "sum", or "none". Describes how to reduce the loss across the batch, use "none" to get the loss for each match.

    Returns
    -------
    classification_loss (tensor):
        The epipolar classification loss, either of shape () if reduction is "mean" or "sum", or of shape (M,) if reduction is "none".
    """
    pseudo_gt, visible_epiline_mask = compute_pseudo_gt_corrs(
        pred_conf, fundamental_matrix, epipolar_line_threshold
    )

    if loss_type == "cross_entropy":
        loss = F.binary_cross_entropy(pred_conf, pseudo_gt, reduction="none")
    else:
        raise NotImplementedError("Unknown loss type: {}".format(loss_type))

    # don't compute the loss if the epipolar line does not appear in image1
    loss = loss * visible_epiline_mask

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()


class EpipolarClassificationLoss(nn.Module):
    def __init__(
        self,
        epipolar_line_threshold: float = sqrt(2.0),
        loss_type: Literal["cross_entropy"] = "cross_entropy",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """A module wrapper for computing the epipolar classification loss.

        Parameters
        ----------
        epipolar_line_threshold (float):
            The threshold for the distance of a pixel to the epipolar line to be considered to be part of the epipolar line. By default sqrt(2.0).

        loss_type (str):
            One of "cross_entropy". Describes the type of classification loss to use. By default "cross_entropy".

        reduction (str):
            One of "mean", "sum", or "none". Describes how to reduce the loss across the batch, use "none" to get the loss for each match.

        """
        super().__init__()
        self.epipolar_line_threshold = epipolar_line_threshold
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self, pred_conf: torch.Tensor, fundamental_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute the epipolar classification loss for a batch of predicted match confidences and fundamental matrices.

        Parameters
        ----------
        pred_conf (batch_size x height0 x width0 x height1 x width1 tensor):
            For each pixel (patch) in image 0, this gives a confidence matrix for the probability that the respective pixel (patch) in image 1 is the matching pixel (patch). If any values are outside the range [0, 1], a dual-softmax will be applied.

        fundamental_matrix (batch_size x 3 x 3 tensor):
            The fundamental matrix from 0 to 1 for the respective image pair.

        Returns
        -------
        classification_loss (tensor):
            The epipolar classification loss, either of shape () if self.reduction is "mean" or "sum", or of shape (M,) if self.reduction is "none".
        """
        return epipolar_classification_loss(
            pred_conf,
            fundamental_matrix,
            self.epipolar_line_threshold,
            self.loss_type,
            self.reduction,
        )
