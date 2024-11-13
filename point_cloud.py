import os
import open3d as o3d
import numpy as np
np.float, np.int = np.float64, np.int_

import argparse
from PIL import Image
import skvideo
skvideo.setFFmpegPath('/opt/homebrew/Cellar/ffmpeg/7.1_3/bin/')
import skvideo.io

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    
    Parameters
    ----------
    quaternion : list or array-like
        A list or array-like of four elements representing the quaternion (qx, qy, qz, qw).
    
    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix.
    
    Raises
    ------
    ValueError
        If the input quaternion does not have four elements.
    """
    if len(quaternion) != 4:
        raise ValueError("Quaternion must have four elements (qx, qy, qz, qw).")

    qx, qy, qz, qw = quaternion

    # Calculate rotation matrix components
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx * qy - qz * qw)
    r13 = 2 * (qx * qz + qy * qw)

    r21 = 2 * (qx * qy + qz * qw)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy * qz - qx * qw)

    r31 = 2 * (qx * qz - qy * qw)
    r32 = 2 * (qy * qz + qx * qw)
    r33 = 1 - 2 * (qx**2 + qy**2)

    # Assemble the rotation matrix
    rotation_matrix = np.array([[r11, r12, r13],
                                 [r21, r22, r23],
                                 [r31, r32, r33]])
    return rotation_matrix

def get_intrinsics(data):
    fx = data['intrinsics'][0, 0]
    fy = data['intrinsics'][1, 1]
    cx = data['intrinsics'][0, 2]
    cy = data['intrinsics'][1, 2]
    intrinsics_scaled = np.array([[fx * DEPTH_WIDTH / 1920, 0.0, cx * DEPTH_WIDTH / 1920],
                                   [0.0, fy * DEPTH_HEIGHT / 1440, cy * DEPTH_HEIGHT / 1440],
                                   [0.0, 0.0, 1.0]])
    return intrinsics_scaled  # Return as a numpy array

def accumulate_point_cloud(pc, rgbd, intrinsics, T_CW):
    rgb, depth = rgbd
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), 
        o3d.geometry.Image(depth),
        depth_scale=1.0, 
        depth_trunc=MAX_DEPTH,
        convert_rgb_to_intensity=False
    )
    return pc + o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, 
                                                                o3d.camera.PinholeCameraIntrinsic(width=DEPTH_WIDTH, height=DEPTH_HEIGHT, 
                                                                                                   fx=intrinsics[0, 0], 
                                                                                                   fy=intrinsics[1, 1], 
                                                                                                   cx=intrinsics[0, 2], 
                                                                                                   cy=intrinsics[1, 2]), 
                                                                extrinsic=T_CW)

def point_clouds(args, data):
    intrinsics = get_intrinsics(data)
    pc = o3d.geometry.PointCloud()
    rgb_path = os.path.join(args.path, 'rgb.mp4')
    video = skvideo.io.vreader(rgb_path)

    for i, (T_WC, rgb) in enumerate(zip(data['poses'], video)):
        if i % args.every != 0:
            continue
        print(f"Point cloud {i}", end="\r")
        T_CW = np.linalg.inv(T_WC)
        confidence = np.array(Image.open(os.path.join(args.path, 'confidence', f'{i:06}.png')))
        
        depth_path = data['depth_frames'][i]
        depth_m = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
        depth_m[confidence < args.confidence] = 0.0

        rgbd = (np.array(Image.fromarray(rgb).resize((DEPTH_WIDTH, DEPTH_HEIGHT))), depth_m)
    
        pc = accumulate_point_cloud(pc, rgbd, intrinsics, T_CW)

    return [pc]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--every', type=int, default=10)
    parser.add_argument('--confidence', type=int, default=0) # 0, 1, or 2
    args = parser.parse_args()

    intrinsics = np.loadtxt(os.path.join(args.path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(args.path, 'odometry.csv'), delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        # timestamp, frame, x, y, z, qx, qy, qz, qw
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = quaternion_to_rotation_matrix(quaternion)
        T_WC[:3, 3] = position
        poses.append(T_WC)
    depth_dir = os.path.join(args.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]


    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Interactive 3D Visualization", width=800, height=600)

    for point_cloud in point_clouds(args, { 'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames }):
        vis.add_geometry(point_cloud)

    vis.get_render_option().point_size = 15.0
    vis.get_render_option().mesh_show_back_face = True

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()