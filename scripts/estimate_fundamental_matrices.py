"""Outline of a script to estimate the fundamental matrices for a dataset of image pairs, which can then be used to fine-tune a pixel correspondence estimator. Since the specifics of this script depend very heavily on the exact dataset and model used, it is likely to need some significant adjustments to apply to a given setting."""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# hyperparameters for the fundamental matrix estimation
MIN_MATCHES = 40
MIN_KEYPOINTS = 150
PIXEL_THR = 0.5
CONFIDENCE = 0.99999


def setup_dataset(args: argparse.Namespace) -> Dataset:
    """Function wrapper to setup the dataset that the fundamental matrices are estimated for.

    Parameters
    ----------
    args (argparse.Namespace):
        any command-line arguments that may be needed.

    Returns
    -------
    Dataset:
        the dataset that the fundamental matrices are estimated for. Given an index, this dataset should return a pair of images to be used by the pre-trained pixel matcher. An example for a possible format can be found in the EuRoCMAV dataset class in test_eurocmav.py.
    """

    raise NotImplementedError("Please implement this function for your chosen dataset.")


def setup_model(args: argparse.Namespace) -> nn.Module:
    """Function wrapper to setup the correspondence estimator model that estimates fundamental matrices.

    Parameters
    ----------
    args (argparse.Namespace):
        any command-line arguments that may be needed.

    Returns
    -------
    nn.Module:
        the pixel correspondence estimator model. The forward() function of this model should have / is assumed to have signature (batch of image pairs as returned by the dataset) -> (list of length batch_size of n_matches x 2 tensors of pixel coordinates in the first image, list of length batch_size of n_matches x 2 tensors of pixel coordinates in the second image).
    """

    raise NotImplementedError(
        "Please implement this function for your chosen pretrained pixel-matching model."
    )


def estimate_fundamental_matrices(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    model = setup_model(args)
    model.eval()
    torch.autograd.set_grad_enabled(False)

    dataset = setup_dataset(args)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    n_estimated_fundamental_matrices = 0
    for batch_idx, image_pairs in tqdm(
        enumerate(dataloader), "Estimating fundamental matrices..."
    ):
        # get pixel correspondence estimates
        batch_pixel_coords0, batch_pixel_coordds1 = model(image_pairs)

        # estimate fundamental matrices
        for i, (pixel_coords0, pixel_coords1) in enumerate(
            zip(batch_pixel_coords0, batch_pixel_coordds1)
        ):
            fundamental_mat, match_mask = cv2.findFundamentalMat(
                pixel_coords0.cpu().numpy(),
                pixel_coords1.cpu().numpy(),
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=PIXEL_THR,
                confidence=CONFIDENCE,
            )

            if (
                fundamental_mat is None
                or fundamental_mat.shape != (3, 3)
                or match_mask.sum() < MIN_MATCHES
            ):
                success = False
            else:
                success = True

            if success:
                n_estimated_fundamental_matrices += 1

                # save the estimated fundamental matrix
                # TODO the user will probably want to save the fundamental matrix in a format and file structure that is more convenient for their dataset.
                output_path = os.path.join(
                    args.output_dir, f"{batch_idx * args.batch_size + i}.npy"
                )
                np.save(output_path, fundamental_mat)

    print("Done!")
    print(
        f"Successfully estimated {n_estimated_fundamental_matrices}/{len(dataset)} fundamental matrices."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate fundamental matrices from a dataset of image pairs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to estimate fundamental matrices for.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory to save the estimated fundamental matrices.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to use for estimating fundamental matrices.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers to use for loading data.",
    )
    # TODO add any additional arguments that may be needed

    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)

    # Estimate the fundamental matrices
    estimate_fundamental_matrices(args)
