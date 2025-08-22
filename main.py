import os
import open3d as o3d
from config import json_path, pcds_dir, idx
from visualizer_utils import load_one_scene


def refresh():
    global o3d_pcd, idx

    print("Current frame: ", idx)
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    pcd, color = load_one_scene(frames[idx])

    vis.remove_geometry(o3d_pcd, reset_bounding_box=False)
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    o3d_pcd.colors = o3d.utility.Vector3dVector(color)
    vis.add_geometry(o3d_pcd, reset_bounding_box=False)

    ctr = vis.get_view_control()
    if json_path:
        parameters = o3d.io.read_pinhole_camera_parameters(json_path)
        ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        ctr.convert_from_pinhole_camera_parameters(cam)

    vis.update_renderer()


def next_frame(_):
    global idx
    idx = (idx + 1) % len(frames)
    refresh()
    return False


def prev_frame(_):
    global idx
    idx = (idx - 1) % len(frames)
    refresh()
    return False


# Load all bin paths
frames = sorted(f[:-4] for f in os.listdir(pcds_dir) if f.endswith(".bin"))

# Load first scene
pts0, col0 = load_one_scene(frames[idx])
o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts0))
o3d_pcd.colors = o3d.utility.Vector3dVector(col0)

# Visualization
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Visualizer")
vis.add_geometry(o3d_pcd)
vis.register_key_callback(ord("N"), next_frame)
vis.register_key_callback(ord("B"), prev_frame)
vis.run()
vis.destroy_window()
