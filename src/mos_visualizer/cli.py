import argparse
from .config import load_config
from .viewer import Viewer


def main():
    parser = argparse.ArgumentParser(description="MOS Visualizer")
    parser.add_argument("--config", "-c", required=True, help="YAML 설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    Viewer(cfg).run()
