from typing import Literal

import torch
import torch.nn as nn
from kornia.geometry import epipolar


def epipolar_regression_loss(
    match_pts0: torch.Tensor,
    match_pts1: torch.Tensor,
    fundamental_matrix: torch.Tensor,
    reduce: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Compute the epipolar regression loss for a batch of matches (can be from multiple images). This is the Euclidean (L2) distance of the predicted match in image 1 to the epipolar line induced in image 1 by the match in image 0 and the respective fundamental matrix.

    Parameters
    ----------
    match_pts0 (M x 2 tensor):
        The points in image 0 that the respective match belongs to. In units of image pixels.

    match_pts1 (M x 2 tensor):
        The points in image 1 that the respective match belongs to. In units of image pixels.

    fundamental_matrix (M x 3 x 3 tensor):
        The fundamental matrix from 0 to 1 for the image pair that the respective match belongs to.

    reduce (str):
        One of "mean", "sum", or "none". Describes how to reduce the loss across the batch, use "none" to get the loss for each match.

    Returns
    -------
    regression_loss (tensor):
        The epipolar regression loss, either of shape () if reduce is "mean" or "sum", or of shape (M,) if reduce is "none".
    """
    assert match_pts0.shape == match_pts1.shape

    dist_0epiline_to_1points = epipolar.left_to_right_epipolar_distance(
        match_pts0.unsqueeze(1), match_pts1.unsqueeze(1), fundamental_matrix
    ).squeeze(1)

    if reduce == "mean":
        loss = torch.mean(dist_0epiline_to_1points)
    elif reduce == "sum":
        loss = torch.sum(dist_0epiline_to_1points)
    elif reduce == "none":
        loss = dist_0epiline_to_1points

    return loss


class EpipolarRegressionLoss(nn.Module):
    def __init__(self, reduce: Literal["mean", "sum", "none"] = "mean"):
        """A module wrapper for computing the epipolar regression loss.

        Parameters
        ----------
        reduce (str):
            One of "mean", "sum", or "none". Describes how to reduce the loss across the batch, use "none" to get the loss for each match.
        """
        super().__init__()
        self.reduce = reduce

    def forward(
        self,
        match_pts0: torch.Tensor,
        match_pts1: torch.Tensor,
        fundamental_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the epipolar regression loss for a batch of matches (can be from multiple images). This is the Euclidean (L2) distance of the predicted match in image 1 to the epipolar line induced in image 1 by the match in image 0 and the respective fundamental matrix.

        Parameters
        ----------
        match_pts0 (M x 2 tensor):
            The points in image 0 that the respective match belongs to. In units of image pixels.

        match_pts1 (M x 2 tensor):
            The points in image 1 that the respective match belongs to. In units of image pixels.

        fundamental_matrix (M x 3 x 3 tensor):
            The fundamental matrix from 0 to 1 for the image pair that the respective match belongs to.

        Returns
        -------
        regression_loss (tensor):
            The epipolar regression loss, either of shape () if reduce is "mean" or "sum", or of shape (M,) if reduce is "none".
        """
        return epipolar_regression_loss(
            match_pts0, match_pts1, fundamental_matrix, self.reduce
        )
