#!/usr/bin/env python3
"""Export Peregrine HDF5 layers as JPEG + YOLO‑v8 polygon labels."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
from src.config import CLASS_ID_SPATTER, CLASS_ID_STREAK, get_dataset_path
from src.utils.yolo_segment import load_hdf5_slice
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data"
CAMERA_PATH = "slices/camera_data/visible/0"
MASK_CHANNELS: dict[str, int] = {
    "spatter": CLASS_ID_SPATTER,
    "streak": CLASS_ID_STREAK,
}
YOLO_CLASS_ID = {"spatter": 0, "streak": 1}
JPEG_PARAMS = (cv2.IMWRITE_JPEG_QUALITY, 90)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger("export_yolo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def contours_to_line(cnt: np.ndarray, cls: int, h: int, w: int) -> str:
    pts = cnt.squeeze(1).astype(np.float32)
    pts[:, 0] /= w
    pts[:, 1] /= h
    return f"{cls} " + " ".join(f"{c:.6f}" for c in pts.ravel())


def write_layer(
    idx: int,
    *,
    img: np.ndarray,
    masks: dict[str, np.ndarray],
    h: int,
    w: int,
    img_dir: Path,
    lbl_dir: Path,
) -> None:
    cv2.imwrite(str(img_dir / f"{idx:06d}.jpg"), img, JPEG_PARAMS)

    lines: list[str] = []
    for name, m in masks.items():
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cls_id = YOLO_CLASS_ID[name]
        lines.extend(contours_to_line(c, cls_id, h, w) for c in cnts if len(c) >= 3)

    (lbl_dir / f"{idx:06d}.txt").write_text("\n".join(lines))


def export(
    build_key: str,
    *,
    val_split: float = 0.0,
    workers: int | None = None,
    out_root: Path | None = None,
) -> None:
    data_path = validate_and_get_data_path(build_key)
    if not data_path:
        return

    out_root, img_train, img_val, lbl_train, lbl_val = prepare_output_directories(
        build_key, out_root
    )

    with h5py.File(data_path, "r") as h5:
        img_ds, mask_dss, h, w = validate_hdf5_file(h5, data_path)
        if not img_ds or not mask_dss:
            return

        export_layers(
            img_ds,
            mask_dss,
            h,
            w,
            val_split,
            workers,
            img_train,
            img_val,
            lbl_train,
            lbl_val,
        )

    logger.info("Done – images & labels stored in %s", out_root)


def validate_and_get_data_path(build_key: str) -> Path | None:
    data_path_str = get_dataset_path(build_key)
    if data_path_str is None:
        logger.error("Unknown build key: %s", build_key)
        return None
    data_path = Path(data_path_str)
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        return None
    return data_path


def prepare_output_directories(build_key: str, out_root: Path | None) -> tuple:
    out_root = (out_root or DEFAULT_OUT) / build_key
    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)
    return out_root, img_train, img_val, lbl_train, lbl_val


def validate_hdf5_file(h5: h5py.File, data_path: Path) -> tuple:
    if CAMERA_PATH not in h5:
        logger.error("Camera path %s missing in %s", CAMERA_PATH, data_path)
        return None, None, None, None
    img_ds = h5[CAMERA_PATH]
    h, w = img_ds.shape[1:]

    mask_dss: dict[str, h5py.Dataset] = {}
    for name, ch in MASK_CHANNELS.items():
        key = f"slices/segmentation_results/{ch}"
        if key in h5:
            mask_dss[name] = h5[key]
        else:
            logger.warning("Mask channel %s (id %d) missing – skipping", name, ch)
    if not mask_dss:
        logger.error("No mask datasets found – aborting")
        return None, None, None, None

    return img_ds, mask_dss, h, w


def export_layers(
    img_ds: h5py.Dataset,
    mask_dss: dict[str, h5py.Dataset],
    h: int,
    w: int,
    val_split: float,
    workers: int | None,
    img_train: Path,
    img_val: Path,
    lbl_train: Path,
    lbl_val: Path,
) -> None:
    n_layers = img_ds.shape[0]
    logger.info("Exporting %d layers (%dx%d)", n_layers, w, h)

    rng = random.Random(42)
    futures: list = []
    n_workers = workers or os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=min(n_workers, 8)) as pool:
        for i in range(n_layers):
            is_val = rng.random() < val_split
            img_dir = img_val if is_val else img_train
            lbl_dir = lbl_val if is_val else lbl_train

            img = load_hdf5_slice(str(img_ds.file.filename), i, CAMERA_PATH)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            masks_np = {n: ds[i].astype(np.uint8) for n, ds in mask_dss.items()}
            futures.append(
                pool.submit(
                    write_layer,
                    i,
                    img=img,
                    masks=masks_np,
                    h=h,
                    w=w,
                    img_dir=img_dir,
                    lbl_dir=lbl_dir,
                )
            )

        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Writing layers",
            unit="layer",
            dynamic_ncols=True,
        ):
            pass


if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="Export YOLO‑v8 training data from Peregrine HDF5"
    )
    cli.add_argument("build_key", help="Dataset key (e.g. tcr_phase1_build2)")
    cli.add_argument("--out_root", type=Path, help="Destination root for data/")
    cli.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Fraction of layers to send to validation (0‑1)",
    )
    cli.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="#threads for JPEG / label writes",
    )
    args = cli.parse_args()

    export(
        args.build_key,
        val_split=args.val_split,
        workers=args.workers,
        out_root=args.out_root,
    )
