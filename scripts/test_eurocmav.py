import argparse
import os
from bisect import bisect
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric
from tqdm import tqdm

# NOTE The camera matrix is the same for all cam0 images from all environments in EuRoC-MAV
INTRINSIC = torch.tensor(
    [[458.654, 0.0, 367.215 - 56], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]]
)  # this is for 640 x 480 images, the original dimensions were 752 x 480, so I subtract 56 to account for cropping


def invert_se3(T):
    """Invert an SE(3) transformation matrix."""
    assert T.shape[-2:] == (4, 4), "T must be of shape (..., 4, 4)"

    rot = T[..., :3, :3]
    trans = T[..., :3, 3]

    if type(T) == torch.Tensor:
        inv_T = torch.zeros_like(T)
        inv_rot = rot.transpose(-1, -2)
        inv_trans = torch.einsum("...ij,...j->...i", -inv_rot, trans)

    else:  # numpy
        inv_T = np.zeros_like(T)
        inv_rot = np.swapaxes(rot, -1, -2)
        inv_trans = np.einsum("...ij,...j->...i", -inv_rot, trans)

    inv_T[..., :3, :3] = inv_rot
    inv_T[..., :3, 3] = inv_trans
    inv_T[..., 3, 3] = 1.0

    return inv_T


class EuRoCMAV(nn.utils.data.Dataset):
    def __init__(self, dataset_dir: str, scene_names_file: str, split: str = "test"):
        super().__init__()

        self.root_dir = dataset_dir
        npz_files = np.loadtxt(scene_names_file, dtype=str)
        self.scene_names = [f.split(".")[0] for f in npz_files]

        self.cum_pairs_per_scene = [0]
        self.img_timestamps = []
        for npz_file in npz_files:
            npz_path = os.path.join(self.root_dir, "index", split, npz_file)

            data = np.load(npz_path)
            self.img_timestamps.append(data["img_idxs"])
            self.cum_pairs_per_scene.append(
                self.cum_pairs_per_scene[-1] + len(data["img_idxs"])
            )

    def __len__(self):
        return self.cum_pairs_per_scene[-1]

    def __getitem__(self, idx):

        scene_idx = bisect(self.cum_pairs_per_scene, idx) - 1
        scene_name = self.scene_names[scene_idx]
        timestamp0, timestamp1 = self.img_timestamps[scene_idx][idx]

        img_path0 = os.path.join(
            self.root_dir, scene_name, "undistorted_imgs", f"{timestamp0}.png"
        )
        img_path1 = os.path.join(
            self.root_dir, scene_name, "undistorted_imgs", f"{timestamp1}.png"
        )

        image0 = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

        # crop to 640 x 480
        image0 = image0[:, 56:-56]
        image1 = image1[:, 56:-56]

        image0 = torch.from_numpy(image0).float()[None] / 255
        image1 = torch.from_numpy(image1).float()[None] / 255

        image_pair = torch.stack([image0, image1], dim=0)

        ## get relative pose
        pose_path0 = os.path.join(
            self.root_dir, scene_name, "pose", f"{timestamp0}.txt"
        )
        pose_path1 = os.path.join(
            self.root_dir, scene_name, "pose", f"{timestamp1}.txt"
        )

        cam2world0 = np.loadtxt(pose_path0, delimiter=" ")
        cam2world1 = np.loadtxt(pose_path1, delimiter=" ")
        world2cam1 = invert_se3(cam2world1)

        gt_pose = np.matmul(world2cam1, cam2world0)

        return {"image_pair": image_pair, "gt_pose": gt_pose}


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


def compute_symmetrical_epipolar_errors(
    kpts0: np.ndarray, kpts1: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """Compute the symmetrical epipolar errors between two sets of keypoints given a relative pose.

    Parameters
    ----------
    kpts0 (n_matches x 2 np.ndarray):
        The pixel coordinates of the keypoints in the first image.

    kpts1 (n_matches x 2 np.ndarray):
        The pixel coordinates of the keypoints in the second image.

    pose (4 x 4 np.ndarray):
        The (ground truth) relative transformation matrix from the first to the second camera frame.

    Returns
    -------
    n_matches np.ndarray:
        The symmetrical epipolar errors between the two sets of keypoints given the relative pose.

    """
    # compute the essential matrix
    Tx = numeric.cross_product_matrix(pose[:3, 3])
    E_mat = Tx @ pose[:3, :3]

    pts0 = (kpts0 - INTRINSIC[[0, 1], [2, 2]][None]) / INTRINSIC[[0, 1], [0, 1]][None]
    pts1 = (kpts1 - INTRINSIC[[0, 1], [2, 2]][None]) / INTRINSIC[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E_mat.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E_mat  # [N, 3]

    d = p1Ep0**2 * (
        1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
        + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2)
    )  # N

    return d


def estimate_pose(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    ransac_thresh: float = 0.5,
    ransac_conf: float = 0.99999,
) -> np.ndarray:
    """Estimate the relative pose between two images given a set of pixel correspondences.

    Parameters
    ----------
    kpts0 (n_matches x 2 np.ndarray):
        The pixel coordinates of the keypoints in the first image.

    kpts1 (n_matches x 2 np.ndarray):
        The pixel coordinates of the keypoints in the second image.

    ransac_thresh (float, optional):
        The threshold for RANSAC to count a match as an inlier, in pixels, by default 0.5

    ransac_conf (float, optional):
        The desired confidence of the RANSAC algorithm, by default 0.99999

    Returns
    -------
    4 x 4 np.ndarray:
        The relative transformation matrix from the first to the second camera frame.
    """
    if len(kpts0) < 5:
        return None

    # normalize keypoints
    kpts0 = (kpts0 - INTRINSIC[[0, 1], [2, 2]][None]) / INTRINSIC[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - INTRINSIC[[0, 1], [2, 2]][None]) / INTRINSIC[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thresh = ransac_thresh / np.mean(
        [INTRINSIC[0, 0], INTRINSIC[1, 1], INTRINSIC[0, 0], INTRINSIC[1, 1]]
    )

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0,
        kpts1,
        np.eye(3),
        threshold=ransac_thresh,
        prob=ransac_conf,
        method=cv2.RANSAC,
    )
    if E is None:
        return None

    # recover pose from E
    best_num_inliers = 0
    ret_transform = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret_transform = np.eye(4)
            ret_transform[:3, :3] = R
            ret_transform[:3, 3] = t[:, 0]
            best_num_inliers = n

    return ret_transform


def relative_pose_error(
    pred_pose: np.ndarray, gt_pose: np.ndarray
) -> Tuple[float, float]:
    """Compute the relative pose error for a predicted pose and a ground truth pose.

    Parameters
    ----------
    pred_pose (4 x 4 np.ndarray):
        The predicted transformation matrix from the first to the second camera frame.

    gt_pose (4 x 4 np.ndarray):
        The ground truth transformation matrix from the first to the second camera frame.

    Returns
    -------
    Tuple[float, float]:
        The rotation error (in degrees) and translation error (in degrees, measuring the angle between the two vectors) between the predicted and ground truth poses.
    """
    pred_t = pred_pose[:3, 3]
    gt_t = gt_pose[:3, 3]
    pred_R = pred_pose[:3, :3]
    gt_R = gt_pose[:3, :3]

    # angle error between 2 rotation matrices
    cos = (np.trace(np.dot(pred_R.T, gt_R)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    # angle error between 2 vectors
    norm = np.linalg.norm(pred_t) * np.linalg.norm(gt_t)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(pred_t, gt_t) / norm, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity

    return R_err, t_err


def error_auc(errors: list[float], thresholds: list[float]) -> dict[str, float]:
    """Compute the Area Under Curve (AUC) given a list of errors and a list of thresholds.

    Parameters
    ----------
    errors (list[float]):
        A list of errors for different samples.

    thresholds (list[float]):
        A list of the error thresholds for which to compute the Area under the Precision-Recall Curve.

    Returns
    -------
    dict[str, float]:
        A dictionary containing the AUC values for each threshold, in the format "auc@{threshold}": AUC.
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}


def epi_error_prec(errors: list[float], thresholds: list[float]) -> dict[str, float]:
    """Compute the precision given a list of errors for individual image pairs and a list of thresholds.

    Parameters
    ----------
    errors (list[n_matches np.ndarray]):
        A list of arrays containing the epipolar errors for the matched keypoints in different image pairs.

    thresholds (list[float]):
        A list of the error thresholds for which to compute the precision.

    Returns
    -------
    dict[str, float]:
        A dictionary containing the precision values for each threshold, in the format "prec@{threshold}": Precision.
    """

    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)

    return {f"prec@{thr:.0e}": prec for thr, prec in zip(thresholds, precs)}


def compute_metrics(
    args: argparse.Namespace,
    batch_kpts0: list[torch.Tensor],
    batch_kpts1: list[torch.Tensor],
    gt_poses: torch.Tensor,
    metrics: dict,
) -> dict:
    """Compute metrics for the given batch of image pairs and ground truth poses. Update the metrics dictionary with the computed metrics by appending the computed metrics to the corresponding list.

    Parameters
    ----------
    args (argparse.Namespace):
        the command line arguments.

    batch_kpts0 (list[torch.Tensor]):
        A list of n_matches x 2 tensors of pixel coordinates in the first image.

    batch_kpts1 (list[torch.Tensor]):
        A list of n_matches x 2 tensors of pixel coordinates in the second image.

    gt_poses (batch_size x 4 x 4 torch.Tensor):
        The ground truth relative camera poses (4x4 transformation matrices from the first to the second camera frame)

    metrics (dict):
        A str: list dict containing the metrics for different image pairs

    Returns
    -------
    dict:
        The updated metrics dictionary.
    """
    if "epi_errs" not in metrics:
        metrics["epi_errs"] = []
    if "R_errs" not in metrics:
        metrics["R_errs"] = []
    if "t_errs" not in metrics:
        metrics["t_errs"] = []

    for kpts0, kpts1, gt_pose in zip(batch_kpts0, batch_kpts1, gt_poses):
        kpts0 = kpts0.cpu().numpy()
        kpts1 = kpts1.cpu().numpy()
        gt_pose = gt_pose.cpu().numpy()

        # compute epipolar projection errors
        epi_errs = compute_symmetrical_epipolar_errors(kpts0, kpts1, gt_pose)
        metrics["epi_errs"].append(epi_errs)

        # compute relative pose errors
        pred_pose = estimate_pose(kpts0, kpts1, args.ransac_thresh, args.ransac_conf)
        if pred_pose is None:
            metrics["R_errs"].append(np.inf)
            metrics["t_errs"].append(np.inf)
        else:
            R_err, t_err = relative_pose_error(pred_pose, gt_pose)
            metrics["R_errs"].append(R_err)
            metrics["t_errs"].append(t_err)

    return metrics


def aggregate_metrics(args: argparse.Namespace, metrics: dict) -> dict:
    """Take in a dictionary of lists of metrics for different image pairs and compute aggregate metrics. Return the updated dictionary

    Parameters
    ----------
    args (argparse.Namespace):
        The command line arguments.

    metrics (dict):
        A str: list dict containing the metrics for different image pairs. Should at least contain the keys "R_errs" and "t_errs".

    Returns
    -------
    dict:
        The dictionary with the aggregated metrics added.
    """
    # Matching Precision
    metrics.update(epi_error_prec(metrics["epi_errs"], args.epi_error_thresholds))

    # Matching AUC
    pose_errors = np.max(
        np.stack([metrics["R_errs"], metrics["t_errs"]], axis=0), axis=0
    )
    metrics.update(error_auc(pose_errors, args.angular_error_thresholds))

    # Other metrics
    metrics.update(
        {
            f"frac_rot_errs_under_{t}": np.mean(np.array(metrics["R_errs"]) < t)
            for t in args.angular_error_thresholds
        }
    )
    metrics.update(
        {
            f"frac_t_errs_under_{t}": np.mean(np.array(metrics["t_errs"]) < t)
            for t in args.angular_error_thresholds
        }
    )
    metrics.update(
        {
            "mean_rot_err": np.mean(metrics["R_errs"]),
            "median_rot_err": np.median(metrics["R_errs"]),
            "mean_t_err": np.mean(metrics["t_errs"]),
            "median_t_err": np.median(metrics["t_errs"]),
        }
    )

    return metrics


def main(args: argparse.Namespace) -> None:

    dataset = EuRoCMAV(args.dataset_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    model = setup_model(args)
    model.eval()
    torch.autograd.set_grad_enabled(False)

    metrics = {}
    for batch in tqdm(dataloader, desc="Testing model on EuRoC-MAV..."):
        image_pairs = batch["image_pair"]
        gt_poses = batch["gt_pose"]

        batch_pixel_coords0, batch_pixel_coords1 = model(image_pairs)

        metrics = compute_metrics(
            args, batch_pixel_coords0, batch_pixel_coords1, gt_poses, metrics
        )

    metrics = aggregate_metrics(metrics)

    print("Computed metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="euroc-mav/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
