import os
import argparse
import numpy as np
import cv2

from tqdm import tqdm

# NOTE The camera matrix and distortion coefficient are the same for all cam0 images from all environments
# distortion coefficients k1, k2, r1, r2
DIST = [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]

# camera matrix; fx, fy, cx, cy (x to the right, y down)
INTRINSICS = [458.654, 457.296, 367.215, 248.375]


def crop_resize(image, img_size):
    # Image size is (width, height)
    image_shape = image.shape[:2]  # height, width
    target_aspect_ratio = img_size[1] / img_size[0]  # height / width

    # crop to the correct aspect ratio, then resize
    if image_shape[0] / image_shape[1] > target_aspect_ratio:
        # crop the top and bottom
        new_height = int(image_shape[1] * target_aspect_ratio)
        start = (image_shape[0] - new_height) // 2
        end = start + new_height
        new_image = image[start:end, :]
    elif image_shape[0] / image_shape[1] < target_aspect_ratio:
        # crop the left and right
        new_width = int(image_shape[0] / target_aspect_ratio)
        start = (image_shape[1] - new_width) // 2
        end = start + new_width
        new_image = image[:, start:end]
    else:
        new_image = image
    return cv2.resize(new_image, img_size)


def main(args):
    dist_coeffs = np.array(DIST)

    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = INTRINSICS[0]
    camera_matrix[1, 1] = INTRINSICS[1]
    camera_matrix[0, 2] = INTRINSICS[2]
    camera_matrix[1, 2] = INTRINSICS[3]

    dataset_folder = args.data_path
    run_folders = os.listdir(dataset_folder)
    run_folders.sort()

    for run_folder in run_folders:
        if not run_folder[:2] in ["MH", "V1", "V2"]:
            continue
        if "." in run_folder:
            continue

        print(f"Starting to process {run_folder}...")

        run_path = os.path.join(dataset_folder, run_folder)
        image_folder = os.path.join(run_path, "mav0", "cam0", "data")
        image_files = os.listdir(image_folder)
        image_files.sort()
        image_files = [
            os.path.join(image_folder, image_file) for image_file in image_files
        ]

        save_folder = os.path.join(run_path, "undistorted_imgs")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for img_file in tqdm(image_files, desc=f"Undistorting images in {run_folder}"):
            # load image
            image = cv2.imread(img_file).astype(float) / 255
            image = image[:, :, 0]  # grayscale anyways

            # undistort image
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                None,
                camera_matrix,
                (image.shape[1], image.shape[0]),
                cv2.CV_32FC1,
            )
            undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            undistorted_image = crop_resize(
                undistorted_image, (args.img_width, args.img_height)
            )
            cv2.imwrite(
                os.path.join(save_folder, os.path.basename(img_file)),
                undistorted_image * 255,
            )

        print(f"Finished undistorting images in {run_folder}.")

        if not (args.img_height == 480 and args.img_width == 752):
            print(
                f"WARNING: The images were resized, so the intrinsics need to be updated."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./",
        help="Path to dataset folder",
    )
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--img_width", type=int, default=752)
    args = parser.parse_args()
    main(args)
