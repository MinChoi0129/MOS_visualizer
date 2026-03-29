"""Open3D 뷰어 + 키 콜백 로직"""

import os
import numpy as np
import open3d as o3d
from . import cv_display
from .loader import load_scene, load_feature
from .coloring import VIRIDIS_BG


class Viewer:
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.mode == "feature":
            # feature 모드: 단일 npy 파일에서 피쳐맵 viridis 시각화
            pts, col = load_feature(cfg)
            self.bg_color = VIRIDIS_BG
            self.frames = None
        else:
            # pcd/gt/pred/confusion 모드: bin+label 다중 프레임 시각화
            self.frames = sorted(f[:-4] for f in os.listdir(cfg.pcds_dir) if f.endswith(".bin"))
            self.idx = min(cfg.idx, len(self.frames) - 1)
            pts, col = load_scene(cfg, self.frames[self.idx])
            self.bg_color = np.array([1.0, 1.0, 1.0])

        self.pts, self.col = pts, col
        self.o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        self.o3d_pcd.colors = o3d.utility.Vector3dVector(col)

    def _cv_update(self, fid=None, pcd=None, colors=None):
        cv_display.update(imgs_dir=self.cfg.imgs_dir, fid=fid, pcd=pcd, colors=colors)

    def _refresh(self):
        print(f"Frame: {self.idx}")
        cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        pcd, color = load_scene(self.cfg, self.frames[self.idx])
        self.vis.remove_geometry(self.o3d_pcd, reset_bounding_box=False)
        self.o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        self.o3d_pcd.colors = o3d.utility.Vector3dVector(color)
        self.vis.add_geometry(self.o3d_pcd, reset_bounding_box=False)
        ctr = self.vis.get_view_control()
        if self.cfg.json_path and os.path.exists(self.cfg.json_path):
            params = o3d.io.read_pinhole_camera_parameters(self.cfg.json_path)
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        else:
            ctr.convert_from_pinhole_camera_parameters(cam)
        self.vis.update_renderer()
        self._cv_update(fid=self.frames[self.idx], pcd=pcd, colors=color)

    def _next(self, _):
        self.idx = (self.idx + 1) % len(self.frames)
        self._refresh()
        return False

    def _prev(self, _):
        self.idx = (self.idx - 1) % len(self.frames)
        self._refresh()
        return False

    def _on_animation(self, vis):
        cv_display.show()
        return False

    def run(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("MOS Visualizer")
        self.vis.get_render_option().background_color = self.bg_color
        self.vis.add_geometry(self.o3d_pcd)
        self.vis.register_animation_callback(self._on_animation)

        if self.frames is not None:
            self.vis.register_key_callback(ord("N"), self._next)
            self.vis.register_key_callback(ord("B"), self._prev)
            self._cv_update(fid=self.frames[self.idx], pcd=self.pts, colors=self.col)
        else:
            self._cv_update(pcd=self.pts, colors=self.col)

        self.vis.run()
        self.vis.destroy_window()
        cv_display.destroy()
