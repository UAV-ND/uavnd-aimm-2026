#!/usr/bin/env python3
"""
Component test: detector_adapter.GenericDetector (TensorRT + GStreamer camera)

Runs get_target_info() in a loop. Requires Jetson + TensorRT + camera pipeline.

Usage (from repo root):
  python3 test/test_detector_component.py --class black_buoy
  python3 test/test_detector_component.py --class target --duration 30
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import prepend_import_paths, repo_root

prepend_import_paths()

from detector_adapter import GenericDetector


def main():
    parser = argparse.ArgumentParser(description="Test GenericDetector (vision)")
    parser.add_argument(
        "--class",
        dest="cls",
        default="black_buoy",
        choices=["black_buoy", "target", "boat"],
        help="Model class to track",
    )
    parser.add_argument(
        "--engine",
        default="engine.engine",
        help="TensorRT engine path (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run (0 = run until Ctrl+C)",
    )
    args = parser.parse_args()

    root = repo_root()
    engine_path = args.engine
    if not os.path.isabs(engine_path):
        engine_path = os.path.join(root, engine_path)
    if not os.path.isfile(engine_path):
        print("ERROR: engine not found:", engine_path)
        return 1

    print("Loading detector class=%s engine=%s" % (args.cls, engine_path))
    det = GenericDetector(
        engine_path=engine_path,
        target_class=args.cls,
        enable_payload_gpio=False,
    )

    t0 = time.time()
    n = 0
    try:
        while args.duration <= 0 or (time.time() - t0 < args.duration):
            info = det.get_target_info()
            n += 1
            if n % 15 == 0 or (info and info.get("has_target")):
                print("frame=%d info=%s" % (n, info))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        det.close()
        print("Detector closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
