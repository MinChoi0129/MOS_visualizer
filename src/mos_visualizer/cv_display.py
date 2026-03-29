"""OpenCV 표시: 카메라 이미지 + Range View.
무거운 연산(imread, range_view)은 백그라운드 스레드, imshow는 메인 스레드."""

import os
import threading
import cv2
from .range_view import make_range_view

_lock = threading.Lock()
_ready_imgs = {}


def _compute(imgs_dir, fid, pcd, colors):
    results = {}

    if imgs_dir and fid is not None:
        for ext in (".png", ".jpg"):
            path = os.path.join(imgs_dir, f"{fid}{ext}")
            if os.path.exists(path):
                results["Camera"] = cv2.imread(path)
                break

    if pcd is not None:
        results["Range View"] = make_range_view(pcd, colors)

    with _lock:
        _ready_imgs.update(results)


def update(imgs_dir="", fid=None, pcd=None, colors=None):
    """non-blocking: 백그라운드에서 계산 시작"""
    threading.Thread(
        target=_compute, args=(imgs_dir, fid, pcd, colors), daemon=True
    ).start()


def show():
    """메인 스레드에서 호출: 준비된 이미지를 imshow로 표시.
    Open3D animation callback에서 주기적으로 호출."""
    with _lock:
        imgs = dict(_ready_imgs)
        _ready_imgs.clear()

    for name, img in imgs.items():
        cv2.imshow(name, img)

    if imgs:
        cv2.waitKey(1)


def destroy():
    cv2.destroyAllWindows()
