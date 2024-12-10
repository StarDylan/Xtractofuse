import open3d as o3d
import time
import skvideo.io 
import skimage.io
import skimage.transform
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os, subprocess
from matplotlib.ticker import MaxNLocator

from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch


# Make Scikit-Video happy
np.float = np.float64
np.int = np.int_


def video_to_point_cloud(dataset_dir: Path, ransac_matching_threshold: float = 100.0, every: int = 1, max_frames: tuple | None = None, use_icp: bool = True, sample_percentage: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    """
    Convert a video dataset to a ground truth point cloud and an experimental point cloud,
    while calculating various evaluation metrics.

    Args:
        dataset_dir (Path): Path to the dataset

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, dict]: 
            - Accumulated point cloud
            - RGB data for the point cloud
            - Ground truth point cloud
            - Dictionary containing metrics for each frame
    """
    transform_0_to_first = None

    intrinsics = np.genfromtxt(dataset_dir.joinpath("camera_matrix.csv"), delimiter=',')
    poses = get_poses(dataset_dir)

    rgb0, confidence0, depth0 = get_frames(dataset_dir, 0)

    dimensions = rgb0.shape[0], rgb0.shape[1]
    assert_shape(rgb0, (dimensions[0], dimensions[1], 3))
    assert_shape(confidence0, (dimensions[0], dimensions[1]))
    assert_shape(depth0, (dimensions[0], dimensions[1]))

    point_cloud0 = image_to_point_cloud(intrinsics, rgb0, depth0)

    # Initialize point clouds and metrics
    point_cloud_all = (point_cloud0.copy())[confidence0.flatten() >= 2]
    rgb_data_all = (rgb0.copy())[confidence0 >= 2]
    point_cloud_ground_truth = (point_cloud0.copy())[confidence0.flatten() >= 2]

    metrics = {
        "rmse": [],
        "mae": [],
        "chamfer": [],
        "hausdorff": [],
        "precision": [],
        "recall": [],
    }

    print("Processing Frame #1 (for free, since we define it as the origin)")
    initial_rmse = rmse_point_clouds(point_cloud_all, point_cloud_ground_truth)
    initial_mae = mae_point_clouds(point_cloud_all, point_cloud_ground_truth)
    initial_chamfer = chamfer_distance(point_cloud_all, point_cloud_ground_truth)
    initial_hausdorff = hausdorff_distance(point_cloud_all, point_cloud_ground_truth)
    initial_precision, initial_recall = precision_recall_point_clouds(point_cloud_all, point_cloud_ground_truth)

    metrics["rmse"].append(initial_rmse)
    metrics["mae"].append(initial_mae)
    metrics["chamfer"].append(initial_chamfer)
    metrics["hausdorff"].append(initial_hausdorff)
    metrics["precision"].append(initial_precision)
    metrics["recall"].append(initial_recall)

    print(f"Initial RMSE: {initial_rmse:.2f}, Initial MAE: {initial_mae:.2f}, Initial Chamfer: {initial_chamfer:.2f}, Initial Hausdorff: {initial_hausdorff:.2f}, Initial Precision: {initial_precision:.2f}, Initial Recall: {initial_recall:.2f}")

    for i, frame_id in enumerate(range(1, get_total_frames(dataset_dir), every)):
        if max_frames is not None and i >= max_frames:
            print("Reached max frames, stopping")
            break

        print(f"Processing frame #{frame_id+1:04}... Data Load ", end="")
        start_frame = time.time()
        timer()

        # Get new frame
        rgb1, confidence1, depth1 = get_frames(dataset_dir, frame_id)

        dimensions = rgb1.shape[0], rgb1.shape[1]
        assert_shape(rgb1, (dimensions[0], dimensions[1], 3))
        assert_shape(confidence1, (dimensions[0], dimensions[1]))
        assert_shape(depth1, (dimensions[0], dimensions[1]))

        print(f"Done ({timer()}) - Point Cloud ", end="")

        point_cloud1 = image_to_point_cloud(intrinsics, rgb1, depth1)

        print(f"Done ({timer()}) - RANSAC ", end="")

        model_1_to_0 = compute_euclidean_transform_ransac(rgb0, point_cloud0, 
                                                          rgb1, point_cloud1, 
                                                          matching_threshold=ransac_matching_threshold, iterations=1000,
                                                          confidence_image0=confidence0,
                                                          confidence_image1=confidence1)
        
        if model_1_to_0 is None:
            print(f"Done ({timer()}) - RANSAC failed to find a model - Skipping frame")
            continue

        if transform_0_to_first is None:
            transform_1_to_first = model_1_to_0
        else:
            transform_1_to_first = transform_0_to_first + model_1_to_0

        point_cloud1_world = model_1_to_0(point_cloud1)

        if use_icp:
            print(f"Done ({timer()}) - ICP ", end="")

            o3d_pc0 = o3d.geometry.PointCloud()
            o3d_pc1 = o3d.geometry.PointCloud()
            o3d_pc0.points = o3d.utility.Vector3dVector(point_cloud0)
            o3d_pc1.points = o3d.utility.Vector3dVector(point_cloud1)

            icp_result = o3d.pipelines.registration.registration_icp(
                o3d_pc1, o3d_pc0, 0.02, model_1_to_0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            icp_transform = icp_result.transformation
            point_cloud1 = np.asarray(o3d_pc1.transform(icp_transform).points)

        print(f"Done ({timer()}) - Control ", end="")

        point_cloud1_ground_truth = odometry_point_cloud_to_world(point_cloud1, poses[frame_id])
        filtered_point_cloud1_ground_truth = (point_cloud1_ground_truth.copy())[confidence1.flatten() >= 2]

        point_cloud_ground_truth = np.concatenate((point_cloud_ground_truth, filtered_point_cloud1_ground_truth.copy()))

        print(f"Done ({timer()}) - Accumulation ", end="")

        # Filter and sample points
        filtered_point_cloud1 = (point_cloud1_world.copy())[confidence1.flatten() >= 2]
        filtered_video_data1 = (rgb1.copy())[confidence1 >= 2]

        num_points_to_sample = int(len(filtered_point_cloud1) * sample_percentage)
        print("Non GC:", len(filtered_point_cloud1))
        print("GC:", len(filtered_point_cloud1_ground_truth))
        sampled_indices = np.random.choice(len(filtered_point_cloud1), num_points_to_sample, replace=False)
        sampled_point_cloud1 = filtered_point_cloud1[sampled_indices]
        sampled_rgb_data1 = filtered_video_data1[sampled_indices]

        point_cloud_all = np.concatenate((point_cloud_all, sampled_point_cloud1))
        rgb_data_all = np.concatenate((rgb_data_all, sampled_rgb_data1))

        print(f"Done ({timer()}) - Metrics ", end="")

        # Compute Metrics
        rmse_val = rmse_point_clouds(point_cloud1_ground_truth, point_cloud1_world)
        mae_val = mae_point_clouds(point_cloud1_ground_truth, point_cloud1_world)
        chamfer_val = chamfer_distance(point_cloud1_ground_truth, point_cloud1_world)
        hausdorff_val = hausdorff_distance(point_cloud1_ground_truth, point_cloud1_world)
        precision_val, recall_val = precision_recall_point_clouds(point_cloud1_ground_truth, point_cloud1_world)

        metrics["rmse"].append(rmse_val)
        metrics["mae"].append(mae_val)
        metrics["chamfer"].append(chamfer_val)
        metrics["hausdorff"].append(hausdorff_val)
        metrics["precision"].append(precision_val)
        metrics["recall"].append(recall_val)

        print(f"RMSE: {rmse_val:.2f}, MAE: {mae_val:.2f}, Chamfer: {chamfer_val:.2f}, Hausdorff: {hausdorff_val:.2f}, Precision: {precision_val:.2f}, Recall: {recall_val:.2f}")

        print(f"Done ({timer()})", end=" ")
        transform_0_to_first = transform_1_to_first
        point_cloud0 = point_cloud1
        point_cloud0_world = point_cloud1_world
        confidence0 = confidence1
        rgb0 = rgb1

        print(f"--- Frame finished in {time.time() - start_frame:.2f}s")

    return point_cloud_all, rgb_data_all, point_cloud_ground_truth, metrics



####################################################################################################
# Dataset functions
####################################################################################################

def extract_frames(dataset_dir: Path):
    """
    Extract frames from a video file into images so we can index it

    Will not re-extract if the frames are already extracted
    """
    if not dataset_dir.joinpath("rgb/DONE.marker").exists():    
        # Extract frames from video
        # The video might be too large to store uncompressed in memory
        os.makedirs(dataset_dir.joinpath("rgb"), exist_ok=True)
    
        subprocess.run([
            'ffmpeg', '-i', dataset_dir.joinpath("rgb.mp4"), '-vsync', 'vfr', '-q:v', '1', '-start_number', '0',
            os.path.join(dataset_dir.joinpath("rgb"), '%06d.png')
        ])

        dataset_dir.joinpath("rgb/DONE.marker").touch()
    

def validate_dataset(dataset_dir: Path):
    """
    Validate the dataset
    """

    extract_frames(dataset_dir)

    num_rgb = sum(1 for _ in dataset_dir.joinpath("rgb").glob("*.png"))
    num_confidence = sum(1 for _ in dataset_dir.joinpath("confidence").glob("*.png"))
    num_depth = sum(1 for _ in dataset_dir.joinpath("depth").glob("*.png"))

    # Weird case where we skip first frame
    if num_rgb == num_confidence - 1 and num_rgb == num_depth - 1:
        num_confidence -= 1
        num_depth -= 1

    assert num_rgb == num_confidence == num_depth, f"Number of frames mismatch: {num_rgb} != {num_confidence} != {num_depth}"

def get_total_frames(dataset_dir: Path):
    extract_frames(dataset_dir)

    return sum(1 for _ in dataset_dir.joinpath("rgb").glob("*.png"))

def get_frames(dataset_dir: Path, frame_id: int)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the RGB, Confidence and Depth frames at the given frame_id

    RGB is scaled between 0.0 and 1.0

    Confidence and depth are resized (nearest neighbor) to the RGB frame size

    Args:
        dataset_dir (Path): Path to the dataset
        frame_id (int): Frame ID to get
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: RGB, Confidence and Depth frames
    """

    # Ensure frames are extracted
    extract_frames(dataset_dir)

    rgb_frames = sorted(list(dataset_dir.joinpath("rgb").glob("*.png")))
    confidence_frames = sorted(list(dataset_dir.joinpath("confidence").glob("*.png")))
    depth_frames = sorted(list(dataset_dir.joinpath("depth").glob("*.png")))

    num_rgb = len(rgb_frames)
    num_confidence = len(confidence_frames)
    num_depth = len(depth_frames)

    rgb_frame_id = frame_id
    confidence_frame_id = frame_id
    depth_frame_id = frame_id

    # Weird case where we skip first frame
    if num_rgb == num_confidence - 1 and num_rgb == num_depth - 1:
        confidence_frame_id += 1
        depth_frame_id += 1

    assert rgb_frame_id < num_rgb, f"RGB Frame {frame_id} not found in dataset"
    assert confidence_frame_id < num_confidence, f"Confidence Frame {frame_id} not found in dataset"
    assert depth_frame_id < num_depth, f"Depth Frame {frame_id} not found in dataset"

    rgb_frame = skimage.io.imread(rgb_frames[rgb_frame_id])
    rgb_frame = (rgb_frame.astype("float32") / 255.0 )

    confidence_frame = skimage.io.imread(confidence_frames[confidence_frame_id])
    confidence_frame = skimage.transform.resize(confidence_frame, rgb_frame.shape[0:2], order=0)

    depth_frame = skimage.io.imread(depth_frames[depth_frame_id])
    depth_frame = skimage.transform.resize(depth_frame, rgb_frame.shape[0:2], order=0)

    return rgb_frame, confidence_frame, depth_frame

def get_poses(dataset_dir: Path):
    poses = []
    
    odometry = np.loadtxt(os.path.join(dataset_dir, 'odometry.csv'), delimiter=',', skiprows=1)

    for line in odometry:
        position, quaternion = line[2:5], line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = quaternion_to_rotation_matrix(quaternion)
        T_WC[:3, 3] = position
        poses.append(T_WC)

    # change all poses such that the first pose is at the origin
    first_pose = poses[0]
    for i in range(len(poses)):
        poses[i] = np.linalg.inv(first_pose) @ poses[i]

    return poses

####################################################################################################
# Visualization functions
####################################################################################################

def colorize_confidence_map(confidence_image: np.ndarray) -> np.ndarray:
    """
    Colorize the confidence map

    Args:
        confidence (np.ndarray): Confidence map

    Returns:
        np.ndarray: Colorized confidence map
    """
    
    new_rgb_image = np.zeros((confidence_image.shape[0], confidence_image.shape[1], 3))
    confidence_image = skimage.transform.resize(confidence_image, new_rgb_image.shape[0:2], order=0)

    new_rgb_image[..., 0] = (confidence_image == 0)
    new_rgb_image[..., 1] = (confidence_image == 1)
    new_rgb_image[..., 2] = (confidence_image == 2)

    rgb_image = new_rgb_image * 255
    return new_rgb_image

def show_random_sample(dataset_path: Path, n: int = 5, seed: int = 42):
    """
    Show up to n random samples from the dataset
    """

    import random

    random.seed(seed)

    total_frames = get_total_frames(dataset_path)

    random_frames = [random.randint(0, total_frames) for _ in range(0, min(total_frames, n))]

    show_samples(dataset_path, random_frames)

def show_samples(dataset_path: Path, frame_ids: list[int]):
    """
    Show specific samples from the dataset
    """

    fig, axes = plt.subplots(nrows=len(frame_ids), ncols=3)
    fig.set_figwidth(15)
    fig.set_figheight(4 * len(frame_ids))

    for i, frame_id in enumerate(frame_ids):
        rgb, confidence, depth = get_frames(dataset_path, frame_id)

        axes[i, 0].imshow(rgb)
        # axes[i, 0].axis("off")

        axes[i, 1].imshow(colorize_confidence_map(confidence))
        # axes[i, 1].axis("off")

        axes[i, 2].imshow(depth)
        # axes[i, 2].axis("off")


def visualize_point_cloud(point_cloud: np.ndarray, rgb: np.ndarray):
    """
    Visualize the point cloud

    Args:
        point_cloud (np.ndarray): Point cloud
        rgb (np.ndarray): RGB data
    """

    
    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=480, width=640)

    points = point_cloud.shape[0]

    assert_shape(point_cloud, (points, 3))
    assert_shape(rgb, (points, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    vis.add_geometry(pcd)


    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

def visualize_light_glue(dataset_dir: Path, frame_id0: int, frame_id1: int):

    rgb0, _, _ = get_frames(dataset_dir, frame_id0)
    rgb1, _, _ = get_frames(dataset_dir, frame_id1)

    m_kpts0, m_kpts1 =  get_matches(rgb0, rgb1)

    axes = viz2d.plot_images([rgb0, rgb1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
####################################################################################################
# Point Cloud + 3D Functions
####################################################################################################

def compute_euclidean_transform_ransac(image0, point_cloud0, image1, point_cloud1, matching_threshold=300.0, iterations=100, confidence_image0=None, confidence_image1=None)-> skimage.transform.EuclideanTransform | None:
    """
    Estimate the Euclidean Transform between two images using RANSAC

    Args:
        image0 (np.ndarray): Image 0
        point_cloud0 (np.ndarray): Point cloud 0
        image1 (np.ndarray): Image 1
        point_cloud1 (np.ndarray): Point cloud 1
        matching_threshold (float, optional): Matching threshold. Defaults to 300.0.
        iterations (int, optional): Number of RANSAC iterations. Defaults to 100.
        confidence_image0 (np.ndarray, optional): Confidence image 0. Defaults to None.
        confidence_image1 (np.ndarray, optional): Confidence image 1. Defaults to None.

    Returns:
        skimage.transform.EuclideanTransform | None: Euclidean Transform or None if not found
    """
    assert_shape(image0, (None, None, 3))
    
    image0_height = image0.shape[0]
    image0_width = image0.shape[1]
    assert_shape(point_cloud0, (image0_height * image0_width, 3))


    assert_shape(image1, (None, None, 3))

    image1_height = image1.shape[0]
    image1_width = image1.shape[1]

    assert_shape(point_cloud1, (image1_height * image1_width, 3))

    if confidence_image0 is None:
        assert confidence_image1 is None
    else:
        assert_shape(confidence_image0, (image0_height, image0_width))
        assert_shape(confidence_image1, (image0_height, image0_width))


    matched_keypoints0, matched_keypoints1 = get_matches(image0, image1)

    num_matches = len(matched_keypoints0)
    assert_shape(matched_keypoints0, (num_matches, 2))
    assert_shape(matched_keypoints1, (num_matches, 2))

    # Round to nearest pixel
    matched_keypoints0 = np.round(matched_keypoints0.cpu().numpy()).astype(int)
    matched_keypoints1 = np.round(matched_keypoints1.cpu().numpy()).astype(int)

    # Turn (x, y) from image coordinates into index into point cloud
    point_cloud0_indices = matched_keypoints0[:, 0] + matched_keypoints0[:, 1] * image0_width
    point_cloud1_indices = matched_keypoints1[:, 0] + matched_keypoints1[:, 1] * image1_width

    # Filter only the high confidence matches
    if confidence_image0 is not None:
        image0_confidence_mask = confidence_image0.flatten() >= 2
        image1_confidence_mask = confidence_image1.flatten() >= 2

        good_keypoint_matches_mask = (image0_confidence_mask[point_cloud0_indices]) & (image1_confidence_mask[point_cloud1_indices])
        point_cloud0_indices = point_cloud0_indices[good_keypoint_matches_mask]
        point_cloud1_indices = point_cloud1_indices[good_keypoint_matches_mask]

    # Get the 3D points for the matched keypoints
    matched_point_cloud0 = point_cloud0[point_cloud0_indices]
    matched_point_cloud1 = point_cloud1[point_cloud1_indices]

    if len(matched_point_cloud0) < 3:
        return None

    best_model = None
    best_inlier_count = 0

    for _ in range(iterations):

        # Get 3 random indices
        random_indices = np.random.choice(len(matched_point_cloud0), 3)

        model = skimage.transform.EuclideanTransform()
        model.estimate(matched_point_cloud0[random_indices], matched_point_cloud1[random_indices])

        transformed_points = model(matched_point_cloud0)

        # Get inliers under threshold
        error = np.sqrt(np.sum((transformed_points - matched_point_cloud1)**2, axis=1))

        inlier_count = np.count_nonzero(error < matching_threshold)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = model
    
    return best_model

def manual_icp(point_cloud1, point_cloud0, initial_transform, max_iterations=20, tolerance=1e-6):
    """
    Manually perform ICP between two point clouds.

    Args:
        point_cloud1 (np.ndarray): Source point cloud (to be aligned).
        point_cloud0 (np.ndarray): Target point cloud (reference).
        initial_transform (np.ndarray): Initial transformation (from RANSAC).
        max_iterations (int): Maximum number of ICP iterations.
        tolerance (float): Convergence threshold.

    Returns:
        np.ndarray: Final transformed source point cloud.
        np.ndarray: Final transformation matrix.
    """
    # Apply initial transformation
    transformed_points = (initial_transform[:3, :3] @ point_cloud1.T).T + initial_transform[:3, 3]
    
    last_error = float('inf')

    for iteration in range(max_iterations):
        # Step 2: Find correspondences (nearest neighbors)
        distances = np.linalg.norm(point_cloud0[:, None, :] - transformed_points[None, :, :], axis=2)
        nearest_indices = np.argmin(distances, axis=0)
        corresponding_points = point_cloud0[nearest_indices]

        # Step 3: Compute centroids
        centroid_src = transformed_points.mean(axis=0)
        centroid_dst = corresponding_points.mean(axis=0)

        # Step 4: Center the points
        src_centered = transformed_points - centroid_src
        dst_centered = corresponding_points - centroid_dst

        # Step 5: Compute cross-covariance matrix
        W = src_centered.T @ dst_centered

        # Step 6: SVD for rotation matrix
        U, _, VT = np.linalg.svd(W)
        R = VT.T @ U.T
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1  # Ensure a right-handed coordinate system

        # Step 7: Translation vector
        t = centroid_dst - R @ centroid_src

        # Step 8: Apply transformation
        transformed_points = (R @ transformed_points.T).T + t

        # Check for convergence
        mean_error = np.mean(np.linalg.norm(transformed_points - corresponding_points, axis=1))
        if abs(last_error - mean_error) < tolerance:
            break
        last_error = mean_error

    # Combine rotation and translation into final transformation matrix
    final_transform = np.eye(4)
    final_transform[:3, :3] = R
    final_transform[:3, 3] = t

    return transformed_points, final_transform

def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])


def points_2d_to_3d(points_homogenous_2d: np.ndarray, depths: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:

    assert_shape(points_homogenous_2d, (None, 3))

    point_num = len(points_homogenous_2d)

    assert_shape(depths, (point_num, ))
    assert_shape(intrinsics, (3,3))

    # Ray from Point in Image:
    # Ideas from https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
    # And https://stackoverflow.com/questions/68249598/how-to-calculate-the-ray-of-a-camera-with-the-help-of-the-camera-matrix

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depths
    x = (points_homogenous_2d[:,0] - cx) * depths / fx
    y = (points_homogenous_2d[:,1] - cy) * depths / fy

    point_cloud = np.stack([x, y, z], axis=1)


    assert_shape(point_cloud, (None, 3))

    return point_cloud[:, 0:3]

def image_to_point_cloud(intrinsics: np.array, rgb_image: np.array, depth_map: np.array) -> np.array:
    assert_shape(intrinsics, (3,3))

    assert_shape(rgb_image, (None, None, 3))
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]

    assert_shape(depth_map, (None, None))


    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    all_pixels = np.stack([xv.ravel(), yv.ravel(), np.ones_like(xv).ravel()], axis=1)

    # Upscale depth to the same as our RGB image
    upscaled_depth_image = skimage.transform.resize(depth_map, rgb_image.shape[0:2], order=0)

    point_cloud = points_2d_to_3d(all_pixels, upscaled_depth_image.flatten(), intrinsics)

    assert_shape(point_cloud, (None, 3))

    return point_cloud

def odometry_point_cloud_to_world(point_cloud, pose):

    assert_shape(point_cloud, (None, 3))
    assert_shape(pose, (4, 4))

    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    point_cloud_world = (rotation @ point_cloud.T).T + translation

    assert_shape(point_cloud_world, (None, 3))
    return point_cloud_world

def rmse_point_clouds(point_cloud0: np.ndarray, point_cloud1: np.ndarray) -> float:
    num_points = len(point_cloud0)

    assert_shape(point_cloud0, (num_points, 3))
    assert_shape(point_cloud1, (num_points, 3))

    point_euclidean_distance = np.sqrt(((point_cloud0 - point_cloud1) ** 2).sum(axis=1))

    return np.sqrt((point_euclidean_distance**2).mean())

def mae_point_clouds(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE) between two point clouds.

    Args:
        pc1 (np.ndarray): First point cloud.
        pc2 (np.ndarray): Second point cloud.

    Returns:
        float: MAE value.
    """
    if len(pc1) == 0 or len(pc2) == 0:
        return float('inf')
    distances = np.linalg.norm(pc1 - pc2, axis=1)
    return np.mean(np.abs(distances))

tree1 = None
tree2 = None

def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    Args:
        pc1 (np.ndarray): First point cloud.
        pc2 (np.ndarray): Second point cloud.

    Returns:
        float: Chamfer distance value.
    """

    if len(pc1) == 0 or len(pc2) == 0:
        return float('inf')
    global tree1, tree2
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    dist1, _ = tree1.query(pc2, k=1)
    dist2, _ = tree2.query(pc1, k=1)

    return np.mean(dist1) + np.mean(dist2)

def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Compute Hausdorff Distance between two point clouds.

    Args:
        pc1 (np.ndarray): First point cloud.
        pc2 (np.ndarray): Second point cloud.

    Returns:
        float: Hausdorff distance value.
    """

    if len(pc1) == 0 or len(pc2) == 0:
        return float('inf')

    dist1, _ = tree1.query(pc2, k=1)
    dist2, _ = tree2.query(pc1, k=1)

    return max(np.max(dist1), np.max(dist2))

def precision_recall_point_clouds(pc1: np.ndarray, pc2: np.ndarray, threshold: float = 10) -> tuple[float, float]:
    """
    Compute precision and recall between two point clouds based on a distance threshold.

    Args:
        pc1 (np.ndarray): First point cloud (ground truth).
        pc2 (np.ndarray): Second point cloud (reconstructed).
        threshold (float): Distance threshold to consider a point as matched.

    Returns:
        tuple[float, float]: Precision and recall values.
    """

    if len(pc1) == 0 or len(pc2) == 0:
        return 0.0, 0.0

    # Precision: Percentage of points in pc2 close to pc1
    dist2, _ = tree2.query(pc1, k=1)
    precision = np.mean(dist2 < threshold)

    # Recall: Percentage of points in pc1 close to pc2
    dist1, _ = tree1.query(pc2, k=1)
    recall = np.mean(dist1 < threshold)

    return precision, recall


####################################################################################################
# 2D Image Functions
####################################################################################################

device = None
extractor = None
matcher = None
def init_light_glue():
    global device, extractor, matcher

    if device is not None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    print(f"Using {device}")

def get_matches(image0: np.ndarray, image1: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """
    Get matches between two images

    Args:
        image0 (np.ndarray): First image
        image1 (np.ndarray): Second image

    Returns:
        tuple[np.ndarray, np.ndarray]: Matched keypoint locations in image0 and image1
    """
    
    global device, extractor, matcher

    # Ensure it is initialized
    init_light_glue()
    
    torch.set_grad_enabled(False)
    
    width = image0.shape[0]
    height = image0.shape[1]

    assert_shape(image1, (width, height, 3))

    image0 = torch.from_numpy(image0.transpose(2, 0, 1))
    image1 = torch.from_numpy(image1.transpose(2, 0, 1))

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return m_kpts0, m_kpts1

####################################################################################################
# Misc Utility Functions
####################################################################################################

last_time = None
def timer():
    global last_time

    if last_time is None:
        last_time = time.time()
        return "0.00s"
    else: 
        elapsed_time = f"{(time.time() - last_time):.2f}s"
        last_time = time.time()

        return elapsed_time
    

# From
# https://medium.com/@nearlydaniel/assertion-of-arbitrary-array-shapes-in-python-3c96f6b7ccb4
from collections import defaultdict
def assert_shape(x, shape:list):
    """ ex: assert_shape(conv_input_array, [8, 3, None, None]) """
    assert len(x.shape) == len(shape), (x.shape, shape)
    for _a, _b in zip(x.shape, shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, shape)