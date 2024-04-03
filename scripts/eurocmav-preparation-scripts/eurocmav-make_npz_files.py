import argparse
import os

import kornia
import numpy as np
import torch
from einops import repeat
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.geometry.linalg import transform_points
from matplotlib import pyplot as plt
from tqdm import tqdm

# NOTE The camera matrix is the same for all cam0 images from all environments
# camera matrix; fx, fy, cx, cy (x to the right, y down)
INTRINSICS = [458.654, 457.296, 367.215, 248.375]


def invert_se3(batch):
    """Invert a batch of SE3 transformations.

    Parameters
    ----------
    batch (torch.Tensor):
        a batch of SE3 transformations, of shape B x 4 x 4

    Returns
    -------
    torch.Tensor:
        the inverted transformations, of shape B x 4 x 4
    """
    R = batch[..., :3, :3]
    t = batch[..., :3, 3:]
    R_inv = R.transpose(1, 2)
    t_inv = torch.matmul(R_inv, -t)
    return torch.cat((torch.cat((R_inv, t_inv), dim=-1), batch[..., 3:, :]), dim=-2)


def intersect_rays_with_plane(camera, plane_height=0.0, hemisphere_radius=10.0):
    """Given a kornia PinholeCamera object, compute the intersection points of the ray through each pixel with a plane at z=plane_height, and a hemisphere of radius hemisphere_radius. This is a rough approximation of the world, but it's good enough for computing an estimate of the overlap of two images.

    Parameters
    ----------
    camera (PinholeCamera):
        the PinholeCamera object for the image (or batch of images)

    plane_height (float, optional):
        the height of the plane, by default 0.0

    hemisphere_radius (float, optional):
        the radius of the hemisphere in meters, by default 10.0

    Returns
    -------
    B x (H*W) x 3 torch.Tensor:
        for each camera in the batch, the intersection points of the rays through each pixel with the plane, in world coordinates
    """

    pixel_coords = (
        kornia.utils.create_meshgrid(
            int(camera.height.item() / 4),
            int(camera.width.item() / 4),
            normalized_coordinates=False,
        ).to(camera.device())
        * 4
    )  # 1 x H x W x 2
    pixel_coords = repeat(
        pixel_coords, "() h w two -> B (h w) two", B=camera.batch_size
    )  # B x (H*W) x 2

    # get the (normalized) vectors for the ray going through each pixel
    # normalise pixel coordinates to have a unity intrinsic matrix
    pixel_coords -= torch.stack((camera.cx, camera.cy), dim=1).view(-1, 1, 2)
    pixel_coords = pixel_coords / torch.stack((camera.fx, camera.fy), dim=1).view(
        -1, 1, 2
    )

    cam_to_world_matrix = invert_se3(camera.extrinsics)
    camera_height = cam_to_world_matrix[..., 2, 3]

    rays_in_camframe = torch.cat(
        (pixel_coords, torch.ones_like(pixel_coords)[..., :1]), dim=-1
    )  # B x (H*W) x 3
    rays_in_camframe = torch.nn.functional.normalize(rays_in_camframe, dim=-1)

    rays_in_worldframe = torch.einsum(
        "bij,bnj->bni",
        cam_to_world_matrix[..., :3, :3],
        rays_in_camframe,
    )

    if (camera_height <= plane_height).any():
        raise ValueError(
            f"Camera is not above the plane (tz is {camera_height} <= plane_height is {plane_height})"
        )

    intersection_params = torch.empty_like(rays_in_worldframe[..., 2])  # B x (H*W)

    # mask for the rays pointing downwards (by enough -- if the camera is 1m above the plane, can only look 1000m away)
    rays_intersecting_with_plane = rays_in_worldframe[..., 2] < -1e-3

    # compute the intersection points with the plane
    plane_intersection_param = (
        plane_height - camera_height.unsqueeze(dim=1)
    ) / torch.clamp(
        rays_in_worldframe[..., 2], max=-1e-3
    )  # B x (H*W)

    intersection_params[rays_intersecting_with_plane] = plane_intersection_param[
        rays_intersecting_with_plane
    ]

    # compute the intersection points with the hemisphere
    # I take the hemisphere to be located at 1000m distance, centered at x=x_camera, y=y_camera, z=plane_height
    shifted_camera_height = (
        camera_height - plane_height
    )  # in the frame where the hemisphere is centered at the origin
    # shifted translation x and y are zero
    shifted_camera_height = shifted_camera_height.unsqueeze(dim=1)  # B x 1

    hemisphere_intersection_param = (
        torch.sqrt(
            torch.square(rays_in_worldframe[..., 2] * shifted_camera_height)
            - torch.square(rays_in_worldframe[..., 2])
            + hemisphere_radius**2
        )
        - rays_in_worldframe[..., 2] * shifted_camera_height
    )

    intersection_params[~rays_intersecting_with_plane] = hemisphere_intersection_param[
        ~rays_intersecting_with_plane
    ]

    world_points = (
        cam_to_world_matrix[..., :3, 3].view(-1, 1, 3)
        + intersection_params.unsqueeze(-1) * rays_in_worldframe
    )

    return world_points


def compute_overlap(camera1, camera2, plane_height=0.0, hemisphere_radius=10.0):
    """Compute a rough estimate of the overlap (intersection over union) of the two images given their poses and intrinsics. I assume that the world consists of a plane at z=plane_height, surrounded by a hemisphere at a reasonable distance, and compute things from there.

    Parameters
    ----------
    camera1 (PinholeCamera):
        the PinholeCamera object for the first image (or batch of first images)

    camera2 (PinholeCamera):
        the PinholeCamera object for the second image (or batch of second images)

    plane_height (float):
        the height of the plane in the world, in meters. Default is 0.

    hemisphere_radius (float):
        the radius of the hemisphere in the world, in meters. Default is 10.

    Returns
    -------
    overlap (torch.Tensor of length B):
        the estimated overlap score for each of the batch of image pairs. 1 means perfect overlap, 0 means no overlap.
    """

    # It's easiest to just compute the depth values for each image, and then just project each pixel. Otherwise I have to compute the intersection of two general quadrilaterals, which is not trivial.

    points1_in_world = intersect_rays_with_plane(
        camera1, plane_height, hemisphere_radius
    )  # B x (H*W) x 3
    points2_in_world = intersect_rays_with_plane(
        camera2, plane_height, hemisphere_radius
    )  # B x (H*W) x 3

    # can't just use the camera.project() function, because that assumes that the points are in front of the camera, which I need to check
    points1_in_camcoords2 = transform_points(camera2.extrinsics, points1_in_world)
    points1_in_2 = convert_points_from_homogeneous(
        transform_points(camera2.intrinsics, points1_in_camcoords2)
    )  # B x (H*W) x 3

    points2_in_camcoords1 = transform_points(camera1.extrinsics, points2_in_world)
    points2_in_1 = convert_points_from_homogeneous(
        transform_points(camera1.intrinsics, points2_in_camcoords1)
    )  # B x (H*W) x 3

    # check if the points are actually visible in the respective image
    mask1 = (
        # in front of camera
        (points1_in_camcoords2[..., 2] > 0)
        # within x image bounds
        & (points1_in_2[..., 0] > 0)
        & (
            points1_in_2[..., 0]
            < repeat(camera2.width, "() -> b ()", b=camera2.batch_size)
        )
        # within y image bounds
        & (points1_in_2[..., 1] > 0)
        & (
            points1_in_2[..., 1]
            < repeat(camera2.height, "() -> b ()", b=camera2.batch_size)
        )
    )  # B x (H*W)

    mask2 = (
        # in front of camera
        (points2_in_camcoords1[..., 2] > 0)
        # within x image bounds
        & (points2_in_1[..., 0] > 0)
        & (
            points2_in_1[..., 0]
            < repeat(camera1.width, "() -> b ()", b=camera1.batch_size)
        )
        # within y image bounds
        & (points2_in_1[..., 1] > 0)
        & (
            points2_in_1[..., 1]
            < repeat(camera1.height, "() -> b ()", b=camera1.batch_size)
        )
    )  # B x (H*W)

    overlap_score = (mask1.sum(dim=-1) + mask2.sum(dim=-1)) / (
        mask1.shape[-1] + mask2.shape[-1]
    )  # B

    return overlap_score


def main(args):
    device = torch.device(args.device)

    camera_matrix = torch.eye(3).to(device)
    camera_matrix[0, 0] = INTRINSICS[0]
    camera_matrix[1, 1] = INTRINSICS[1]
    camera_matrix[0, 2] = INTRINSICS[2]
    camera_matrix[1, 2] = INTRINSICS[3]

    dataset_path = args.data_path
    trajectory_list = os.listdir(dataset_path)
    trajectory_list.sort()

    with open(os.path.join(dataset_path, "index", "test_list.txt"), "r") as f:
        test_list = f.readlines()
    test_list = [x.strip()[:-4] for x in test_list]
    with open(os.path.join(dataset_path, "index", "train_list.txt"), "r") as f:
        train_list = f.readlines()
    train_list = [x.strip()[:-4] for x in train_list]
    with open(os.path.join(dataset_path, "index", "val_list.txt"), "r") as f:
        val_list = f.readlines()
    val_list = [x.strip()[:-4] for x in val_list]

    for trajectory_folder in trajectory_list:
        if not os.path.isdir(os.path.join(dataset_path, trajectory_folder)):
            continue
        if trajectory_folder[:2] == "MH":
            plane_height = (
                -2.0
            )  # This is extremely approximate, but as a very rough heuristic it should be mostly fine
            hemisphere_radius = 10.0  # roughly fine
        elif trajectory_folder[:2] in ["V1", "V2"]:
            plane_height = 0.0
            hemisphere_radius = 3.0  # a normal room; this is an alright scale
        else:
            continue

        if trajectory_folder in test_list:
            split = "test"
            if args.skip_test:
                continue
        elif trajectory_folder in train_list:
            split = "trainval"
            if args.skip_train:
                continue
        elif trajectory_folder in val_list:
            split = "trainval"
            if args.skip_val:
                continue

        print(f"\nProcessing {trajectory_folder}...")
        trajectory_path = os.path.join(dataset_path, trajectory_folder)

        if args.overlap_range[0] < 0:
            filter_overlap = False
        else:
            filter_overlap = True

        if args.overlap_range[0] > 1:  # use default overlap range
            args.overlap_range = (0.4, 0.8)

        # load all poses
        pose_list = os.listdir(os.path.join(trajectory_path, "pose"))
        pose_list.sort()
        all_img_idxs = torch.tensor([int(x[:-4]) for x in pose_list])

        all_cam_to_world = []
        for posefile in tqdm(pose_list, desc="Loading poses"):
            pose = np.loadtxt(os.path.join(trajectory_path, "pose", posefile))
            pose = torch.from_numpy(pose).to(device).float()
            all_cam_to_world.append(pose)

        all_cam_to_world = torch.stack(all_cam_to_world)
        if torch.min(all_cam_to_world[:, 2, 3]) < plane_height:
            print(f"Plane height {plane_height} is above trajectory!")
            breakpoint()
        all_extrinsics = invert_se3(all_cam_to_world)

        if args.visualise:
            plt.plot(all_cam_to_world[:, 2, 3].cpu().numpy())
            plt.savefig(trajectory_folder + "tz_rand.png")
            plt.cla()
            plt.plot(
                all_cam_to_world[:, 0, 3].cpu().numpy(),
                all_cam_to_world[:, 1, 3].cpu().numpy(),
            )
            plt.savefig(trajectory_folder + "xy_rand.png")
            plt.cla()
            print(
                f"Max z value: {torch.max(all_cam_to_world[:, 2, 3]).item()} at timestamp {all_img_idxs[torch.max(all_cam_to_world[:, 2, 3], dim=0)[1].item()]}"
            )
            print(
                f"Min z value: {torch.min(all_cam_to_world[:, 2, 3]).item()} at timestamp {all_img_idxs[torch.min(all_cam_to_world[:, 2, 3], dim=0)[1].item()]}"
            )

        n_pairs_to_sample = args.n_pairs_per_scene
        max_batch_size = int(min(args.batch_size, len(all_img_idxs)))

        if filter_overlap:
            if (
                n_pairs_to_sample > 0.2 * len(all_img_idxs) * 200
            ) or args.try_all_pairs:  # heuristic for maximum expected number of samples that make things too slow
                try_all_pairs = True
                pair_offset = 0
                batch_idx = 0
            else:
                try_all_pairs = False
        elif n_pairs_to_sample >= len(all_img_idxs) ** 2 or args.try_all_pairs:
            try_all_pairs = True
            n_pairs_to_sample = float("inf")
            pair_offset = 0
            batch_idx = 0
        else:
            try_all_pairs = False

        img_idxs = []
        scores = []
        n_pairs_sampled = 0
        n_failures = 0

        prev_idxs = torch.zeros((0, 2), dtype=torch.long)

        # shuffle all_img_idxs and all_extrinsics to get some more randomness
        randperm = torch.randperm(len(all_img_idxs))
        all_img_idxs = all_img_idxs[randperm]
        all_extrinsics = all_extrinsics[randperm]

        while n_pairs_sampled < n_pairs_to_sample:
            if not try_all_pairs:
                if n_failures > 100:
                    print("Too many failures, continuing to next scene")
                    break
                idxs1 = torch.randint(0, len(all_img_idxs), size=(max_batch_size,))
                idxs2 = torch.randint(0, len(all_img_idxs), size=(max_batch_size,))

                if (
                    len(prev_idxs) > 0
                ):  # check I haven't already sampled this image pair
                    mask = torch.ones_like(idxs1)

                    idxs1_previdxs1 = idxs1.view(1, -1) == prev_idxs[:, 0].view(
                        -1, 1
                    )  # N x B
                    idxs1_previdxs2 = idxs1.view(1, -1) == prev_idxs[:, 1].view(
                        -1, 1
                    )  # N x B
                    idxs2_previdxs1 = idxs2.view(1, -1) == prev_idxs[:, 0].view(
                        -1, 1
                    )  # N x B
                    idxs2_previdxs2 = idxs2.view(1, -1) == prev_idxs[:, 1].view(
                        -1, 1
                    )  # N x B

                    mask = (idxs1_previdxs1 & idxs2_previdxs2) | (
                        idxs1_previdxs2 & idxs2_previdxs1
                    )
                    mask = torch.any(mask, dim=0)

                    idxs1 = idxs1[mask]
                    idxs2 = idxs2[mask]

                prev_idxs = torch.cat(
                    [prev_idxs, torch.stack([idxs1, idxs2], dim=1)], dim=0
                )

            else:
                if pair_offset == len(all_img_idxs):
                    break

                idxs1 = torch.arange(len(all_img_idxs))
                idxs2 = torch.arange(
                    start=pair_offset, end=len(all_img_idxs) + pair_offset
                ) % len(all_img_idxs)

                idxs1 = idxs1[
                    batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size
                ]
                idxs2 = idxs2[
                    batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size
                ]

                batch_idx += 1
                if batch_idx * max_batch_size >= len(all_img_idxs):
                    pair_offset += 1
                    batch_idx = 0

            batch_size = len(idxs1)

            if batch_size == 0:
                continue

            if filter_overlap:
                intrinsics = torch.eye(4).to(device)
                intrinsics[:3, :3] = camera_matrix
                intrinsics = repeat(intrinsics, "c1 c2 -> b c1 c2", b=batch_size)
                extrinsics1 = all_extrinsics[idxs1]
                extrinsics2 = all_extrinsics[idxs2]

                camera1 = PinholeCamera(
                    intrinsics,
                    extrinsics1,
                    args.img_height * torch.ones(1, device=extrinsics1.device),
                    args.img_width * torch.ones(1, device=extrinsics1.device),
                )
                camera2 = PinholeCamera(
                    intrinsics,
                    extrinsics2,
                    args.img_height * torch.ones(1, device=extrinsics1.device),
                    args.img_width * torch.ones(1, device=extrinsics1.device),
                )

                overlap_scores = compute_overlap(
                    camera1, camera2, plane_height, hemisphere_radius
                )

                mask = (overlap_scores >= args.overlap_range[0]) & (
                    overlap_scores <= args.overlap_range[1]
                )
                mask = mask.cpu()
                idxs1 = idxs1[mask]
                idxs2 = idxs2[mask]
                scores.append(overlap_scores[mask])

            n_pairs_sampled += len(idxs1)
            if len(idxs1) == 0 and not try_all_pairs:
                n_failures += 1
            else:
                n_failures = 0

            img_idxs1 = all_img_idxs[idxs1]
            img_idxs2 = all_img_idxs[idxs2]
            img_idxs.append(torch.stack([img_idxs1, img_idxs2], dim=-1))

            if try_all_pairs and n_pairs_to_sample == float("inf"):
                n_checked_pairs = pair_offset * len(all_img_idxs) + batch_idx * int(
                    max(max_batch_size, len(all_img_idxs))
                )
                total_pairs = len(all_img_idxs) ** 2
                print(
                    f"Checked {n_checked_pairs} out of {total_pairs} image pairs",
                    end="\r",
                )
            else:
                print(
                    f"Sampled {n_pairs_sampled} out of {n_pairs_to_sample} image pairs",
                    end="\r",
                )

        if len(img_idxs) == 0:
            print(f"Failed to sample any pairs for {trajectory_folder}.")
            continue

        img_idxs = torch.cat(img_idxs, dim=0).cpu().numpy()[:n_pairs_to_sample]
        if filter_overlap:
            scores = torch.cat(scores, dim=0).cpu().numpy()[:n_pairs_to_sample]
        scene = np.array([trajectory_folder for _ in range(len(img_idxs))])[
            :n_pairs_to_sample
        ]

        save_path = os.path.join(
            dataset_path, "index", split, f"{trajectory_folder}.npz"
        )
        np.savez(save_path, scene=scene, img_idxs=img_idxs, score=scores)

        print(f"Finished {trajectory_folder}. Created {len(img_idxs)} pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overlap_range", type=str, default="2,2")
    parser.add_argument("--n_pairs_per_scene", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=1e3)
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--img_width", type=int, default=752)
    parser.add_argument("--try_all_pairs", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--visualise", action="store_true")
    args = parser.parse_args()

    args.overlap_range = tuple(map(float, args.overlap_range.split(",")))
    main(args)
