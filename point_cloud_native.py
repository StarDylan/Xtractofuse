
import os
import numpy as np
from PIL import Image
import argparse
np.float, np.int = np.float64, np.int_
import open3d as o3d
# from plyfile import PlyData, PlyElement
import subprocess

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    
    subprocess.run([
        'ffmpeg', '-i', video_path, '-vsync', 'vfr', '-q:v', '1', '-start_number', '0',
        os.path.join(frame_dir, '%06d.png')
    ])


def quaternion_to_rotation_matrix(quaternion):
    qx, qy, qz, qw = quaternion
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])

def get_intrinsics(data):
    fx = data['intrinsics'][0, 0]
    fy = data['intrinsics'][1, 1]
    cx = data['intrinsics'][0, 2]
    cy = data['intrinsics'][1, 2]
    return np.array([[fx * DEPTH_WIDTH / 1920, 0, cx * DEPTH_WIDTH / 1920],
                     [0, fy * DEPTH_HEIGHT / 1440, cy * DEPTH_HEIGHT / 1440],
                     [0, 0, 1]])

def create_point_cloud(rgbd, intrinsics, T_CW):
    rgb, depth = rgbd
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    points, colors = [], []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z = depth[v, u]
            if z > 0 and z < MAX_DEPTH:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                point_world = T_CW[:3, :3] @ np.array([x, y, z]) + T_CW[:3, 3]
                points.append(point_world)
                colors.append(rgb[v, u] / 255.0)
    return np.array(points), np.array(colors)

def accumulate_point_cloud(pc, rgbd, intrinsics, T_CW):
    points, colors = create_point_cloud(rgbd, intrinsics, T_CW)
    pc[0].extend(points)
    pc[1].extend(colors)

# def save_point_cloud_to_ply(filename, points, colors):
#     vertices = [(points[i, 0], points[i, 1], points[i, 2],
#              int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255))
#             for i in range(points.shape[0])]
#     ply_data = PlyElement.describe(
#         np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
#         'vertex')
#     PlyData([ply_data]).write(filename)

def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def process_point_clouds(args, data):
    intrinsics = get_intrinsics(data)
    pc = ([], [])
    video_path = os.path.join(args.path, 'rgb.mp4')
    frame_dir = os.path.join(args.path, 'frames')
    
    if not os.path.exists(frame_dir):
        extract_frames(video_path, frame_dir)

    for i, T_WC in enumerate(data['poses']):
        if i % args.every != 0:
            continue
        print(f"Point cloud {i}", end="\r")
        T_CW = np.linalg.inv(T_WC)
        confidence = np.array(Image.open(os.path.join(args.path, 'confidence', f'{i:06}.png')))
        depth_path = data['depth_frames'][i]
        depth_m = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0
        depth_m[confidence < args.confidence] = 0.0
        
        rgb_frame_path = os.path.join(frame_dir, f'{i:06}.png')
        rgb_frame = np.array(Image.open(rgb_frame_path))
        rgbd = (np.array(Image.fromarray(rgb_frame).resize((DEPTH_WIDTH, DEPTH_HEIGHT))), depth_m)
        
        accumulate_point_cloud(pc, rgbd, intrinsics, T_WC)

    points, colors = np.array(pc[0]), np.array(pc[1])
    visualize_point_cloud(points, colors)
    # save_point_cloud_to_ply("output.ply", points, colors)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--every', type=int, default=10)
    parser.add_argument('--confidence', type=int, default=1)
    args = parser.parse_args()

    intrinsics = np.loadtxt(os.path.join(args.path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(args.path, 'odometry.csv'), delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        position, quaternion = line[2:5], line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = quaternion_to_rotation_matrix(quaternion)
        T_WC[:3, 3] = position
        poses.append(T_WC)

    depth_dir = os.path.join(args.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]

    process_point_clouds(args, {'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames})

if __name__ == "__main__":
    main()
 
