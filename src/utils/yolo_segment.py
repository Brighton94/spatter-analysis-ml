"""Helper functions for YOLO-based **instance-segmentation** on build slices."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Make sure the slice is (H,W,3) uint8 RGB."""

    if arr.ndim == 2:  # grayscale
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.uint8)


def load_hdf5_slice(h5_path: str | Path, layer: int, dset: str) -> np.ndarray:
    """Load a single layer → (H,W,3) uint8."""

    with h5py.File(h5_path, "r") as h5:
        img = _ensure_rgb(h5[dset][layer])
    return img


def load_hdf5_stack(h5_path: str | Path, dset: str) -> np.ndarray:
    """Load full stack → (N,H,W,3) uint8."""

    with h5py.File(h5_path, "r") as h5:
        raw = h5[dset][:]
    if raw.ndim == 3:  # (N,H,W)
        raw = np.stack([raw] * 3, axis=-1)
    elif raw.ndim == 4 and raw.shape[-1] == 1:
        raw = np.repeat(raw, 3, axis=-1)
    return raw.astype(np.uint8)


def load_yolo_model(
    weights: str | Path = "yolo11n-seg.pt",  # nano → smaller
    device: str | int | None = 0,  # GPU 0 by default
    verbose: bool = True,
) -> YOLO:
    """Load a **YOLOv11-Segment** checkpoint."""

    if verbose:
        logger.info(f"Loading YOLOv11-Segment model ➜  {weights}  (device={device})")
    return YOLO(str(weights), task="segment", device=device)


def semantic_masks(
    res: Any,
    shape: tuple[int, int],
    class_map: dict[str, int] | None = None,
) -> np.ndarray:
    """Return a (2, H, W) uint8 semantic tensor from YOLO instance masks."""

    if class_map is None:
        class_map = {"spatter": 1, "streak": 2}

    H, W = shape
    spat = masks_to_binary(res, [class_map["spatter"]], (H, W))
    strk = masks_to_binary(res, [class_map["streak"]], (H, W))
    return np.stack([spat, strk], axis=0)


def predict(
    model: YOLO,
    images: np.ndarray | Iterable[np.ndarray],
    imgsz: int = 640,
    conf: float = 0.25,
) -> list[Any]:
    """Run instance-segmentation on one or many images."""

    return model(images, imgsz=imgsz, conf=conf)  # type: ignore[return-value]


def masks_to_binary(
    res: Any,
    class_ids: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Combine YOLO masks for the given class_ids into a single (H,W) uint8 mask."""

    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    for i, cls in enumerate(res.boxes.cls.cpu().numpy().astype(int)):
        if cls in class_ids:
            out = np.maximum(out, res.masks.data[i].cpu().numpy().astype(np.uint8))
    return out


def batch_anomaly_areas(
    results: list[Any],
    class_ids: list[int],
    pixel_mm2: float = 1.0,
) -> np.ndarray:
    """Compute anomaly area (mm²) for each image in a batch."""

    areas = []
    for res in results:
        H, W = res.orig_shape
        mask = masks_to_binary(res, class_ids, (H, W))
        areas.append(mask.sum() * pixel_mm2)
    return np.asarray(areas, dtype=float)


def overlay_results(image: np.ndarray, res: Any) -> np.ndarray:
    """Return RGB image with YOLO masks/boxes drawn."""

    return res.plot(labels=False, boxes=False)  # transparent masks + outlines
