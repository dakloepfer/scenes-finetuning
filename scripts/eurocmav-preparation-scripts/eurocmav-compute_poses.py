import os
import argparse
import numpy as np

from tqdm import tqdm

# NOTE This is the same for all cam0 images from all environments
CAM_TO_BODY = np.array(
    [
        [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
        [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
        [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def find_closest_timestamp(timestamp: float, timestamp_list, verbose=False):
    """find the index of the two closest timestamps in timestamp_list

    Parameters
    ----------
    timestamp (float):
        the query timestamp

    timestamp_list (List[float] or np.ndarray):
        the list of timestamps to search, assumed to be increasing

    Returns
    -------
    int, int:
        the index of the two closest timestamps (one before the query timestamp, one after)
    """
    if timestamp_list[0] > timestamp:
        if verbose:
            print(
                f"Warning: timestamp {timestamp} is smaller than all timestamps in timestamp_list, using the first timestamp {timestamp_list[0]}"
            )
        return 0, 0

    for i in range(len(timestamp_list)):
        if timestamp_list[i] == timestamp:
            return i, i

        elif timestamp_list[i] > timestamp:
            return i - 1, i

    if verbose:
        print(
            f"Warning: timestamp {timestamp} is larger than all timestamps in timestamp_list, using the final timestamp {timestamp_list[-1]}"
        )

    return len(timestamp_list) - 1, len(timestamp_list) - 1


def interpolate_pose(query_timestamp, ground_truth_list, skip_out_of_range=True):
    """calculate the interpolated pose at a given timestamp

    Parameters
    ----------
    query_timestamp (float):
        the query timestamp

    ground_truth_list (N x 8 np.ndarray):
        an array with the columns [timestamp, tx, ty, tz, qw, qx, qy, qz]

    skip_out_of_range (bool):
        Whether to return None for query_timestamps that are out of range of the ground_truth_list timestamps. If False, the first / last pose in the list will be returned.

    Returns
    -------
    interpolated_pose (7D np.ndarray):
        the interpolated pose at the query timestamp
    """

    # find the two closest timestamps
    idx1, idx2 = find_closest_timestamp(query_timestamp, ground_truth_list[:, 0])

    if skip_out_of_range:
        if (idx1 == idx2) and (idx1 == 0 or idx1 == len(ground_truth_list) - 1):
            return None

    timestamp1, timestamp2 = ground_truth_list[idx1, 0], ground_truth_list[idx2, 0]
    pose1, pose2 = ground_truth_list[idx1, 1:], ground_truth_list[idx2, 1:]

    interpolated_pose = pose1 + (pose2 - pose1) * (query_timestamp - timestamp1) / max(
        timestamp2 - timestamp1, 1e-10
    )

    return interpolated_pose


def quaternion_to_rot_matrix(quaternion):
    """The inverse of rot_matrix_to_quaternion(). Computes the rotation matrix for a (batch of) quaternion. Uses method from https://journals.sagepub.com/doi/epub/10.1177/0278364915620033.

    Parameters
    ----------
    quaternion (* x 4 array):
        a batch of quaternions ([0] + [1]i + [2]j + [3]k).

    Returns
    -------
    * x 3 x 3 array:
        a batch of corresponding rotation matrices.
    """
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8, None
    )

    qw = quaternion[..., 0]
    qvec = quaternion[..., 1:]

    skew_symm = np.zeros((*quaternion.shape[:-1], 3, 3))
    skew_symm[..., 0, 1] = -qvec[..., 2]
    skew_symm[..., 0, 2] = qvec[..., 1]
    skew_symm[..., 1, 0] = qvec[..., 2]
    skew_symm[..., 1, 2] = -qvec[..., 0]
    skew_symm[..., 2, 0] = -qvec[..., 1]
    skew_symm[..., 2, 1] = qvec[..., 0]

    identity = np.zeros((*quaternion.shape[:-1], 3, 3))
    identity[..., 0, 0] = 1
    identity[..., 1, 1] = 1
    identity[..., 2, 2] = 1

    rot_matrix = (
        qw * qw * identity
        + 2 * qw * skew_symm
        + np.einsum("...ij,...jk->...ik", skew_symm, skew_symm)
        + np.einsum("...i,...j->...ij", qvec, qvec)
    )

    return rot_matrix


def main(args):
    dataset_path = args.data_path

    folder_list = os.listdir(dataset_path)
    folder_list.sort()

    for folder in folder_list:
        if not folder[:2] in ["MH", "V1", "V2"]:
            continue
        if "." in folder:
            continue

        print(f"Processing {folder}...")

        pose_folder = os.path.join(dataset_path, folder, "pose")
        if not os.path.exists(pose_folder):
            os.makedirs(pose_folder)

        ground_truth = np.loadtxt(
            os.path.join(
                dataset_path, folder, "mav0", "state_groundtruth_estimate0", "data.csv"
            ),
            delimiter=",",
        )
        # get rid of all the columns I don't need
        ground_truth = ground_truth[:, :8]

        with open(
            os.path.join(dataset_path, folder, "mav0", "cam0", "data.csv"), "r"
        ) as f:
            img_list = f.readlines()

        img_list = [line.strip().split(",") for line in img_list[1:]]

        img_list = [[float(x[0]), x[1]] for x in img_list]

        img_file_list = os.listdir(
            os.path.join(dataset_path, folder, "undistorted_imgs")
        )

        for img in tqdm(img_list, desc=f"Calculating poses for {folder}"):
            timestamp, img_name = img

            can_load_pose = False
            existing_pose = None
            if os.path.exists(os.path.join(pose_folder, img_name[:-4] + ".txt")):
                try:
                    existing_pose = np.loadtxt(
                        os.path.join(pose_folder, img_name[:-4] + ".txt")
                    )
                    can_load_pose = True
                except Exception as e:
                    pass

            if img_name not in img_file_list:
                print(
                    f"Warning: {img_name} at not found in undistorted_img folder for {folder}, skipping..."
                )
                continue
            if can_load_pose:
                np.savetxt(
                    os.path.join(pose_folder, img_name[:-4] + ".txt"),
                    existing_pose,
                    header="Camera to World Transformation Matrix",
                )
                img_file_list.remove(img_name)
                continue

            else:
                # remove img_name from img_file_list
                img_file_list.remove(img_name)

            interpolated_pose = interpolate_pose(
                timestamp, ground_truth, skip_out_of_range=True
            )
            if interpolated_pose is None:
                continue

            body_to_world = np.eye(4)
            body_to_world[:3, :3] = quaternion_to_rot_matrix(interpolated_pose[3:])
            if np.isnan(body_to_world).any():
                breakpoint()
            body_to_world[:3, 3] = interpolated_pose[
                :3
            ]  # the camera-to-world transformation

            cam_to_world = body_to_world @ CAM_TO_BODY

            np.savetxt(
                os.path.join(pose_folder, img_name[:-4] + ".txt"),
                cam_to_world,
                header="Camera to World Transformation Matrix",
            )

        if len(img_file_list) > 0:
            print(f"Warning: {len(img_file_list)} images not processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Path to dataset folder",
    )
    args = parser.parse_args()
    main(args)
