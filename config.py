# Set paths (pcds_dir: required, gts_dir: required, preds_dir: optional)
pcds_dir = rf"C:\Working\LAB\2025\MOS(HeliMOS)\Dataset_HeliMOS\helimos_raw\Avia\velodyne"
gts_dir = rf"C:\Working\LAB\2025\MOS(HeliMOS)\Dataset_HeliMOS\helimos_raw\Avia\labels"
preds_dir = rf"path/to/predictions"

# Set index and json path
idx = 1100
json_path = ""

# Set options
OPTIONS = {
    "range_crop": {
        "min": 4.0,
        "max": 500.0,
    },
    "quantize": {
        "on": False,
        "range_x": [-50, 50],
        "range_y": [-50, 50],
        "range_z": [-4, 2],
        "grid_sz": (512, 512, 30),
    },
    "2d": {
        "on": False,
    },
    "confusion_vis": {
        "on": False,
    },
}
