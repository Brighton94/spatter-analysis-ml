from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.feature_engineering import (
    count_pixels_stream,  # memory‑friendly pixel counter
    layer_dataframe,  # targets + selected sensors
)

"""Utilities for constructing a layer‑indexed master dataframe
with all available process‑, temporal‑ and segmentation‑derived
features for a single Concept‑Laser M2 build.

The resulting CSV is saved as ``build2_features.csv`` (configurable).

Usage
-----
python -m src.utils.build_features \
       --h5 "data/2021-04-16 TCR Phase 1 Build 2.hdf5" \
       --out "data/build2_features.csv" [--workers 12]
"""

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ─────────────────────────────────────────────────────────────
# 1.  Layer → dominant part‑ID (parallel exact mode)
# ─────────────────────────────────────────────────────────────


def _modal_part_id(idx: int, h5_path: str) -> tuple[int, int]:
    """Return ``(layer_idx, modal_part_id)`` for one layer."""
    with h5py.File(h5_path, "r") as f:
        layer = f["slices/part_ids"][idx]
    non0 = layer[layer > 0]
    modal = int(np.bincount(non0).argmax()) if non0.size else 0
    return idx, modal


def _layer_to_part_map(h5_path: Path, *, workers: int | None = None) -> pd.Series:
    with h5py.File(h5_path, "r") as f:
        n_layers = f["slices/part_ids"].shape[0]

    workers = workers or (os.cpu_count() or 4)
    logger.info(
        "Computing modal part‑ID using %d worker(s) for %d layers", workers, n_layers
    )

    part_ids: list[int] = [0] * n_layers
    tic = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = (exe.submit(_modal_part_id, i, str(h5_path)) for i in range(n_layers))
        for fut in tqdm(
            as_completed(futures), total=n_layers, desc="modal part‑ID", unit="layer"
        ):
            idx, modal = fut.result()
            part_ids[idx] = modal

    logger.info("Modal part‑ID finished in %.1f min", (time.perf_counter() - tic) / 60)
    return pd.Series(part_ids, name="part_id", dtype="int32").rename_axis("layer")


# ─────────────────────────────────────────────────────────────
# 2.  Broadcast per‑part process parameters to layers
# ─────────────────────────────────────────────────────────────


def _broadcast_process_params(h5_path: Path, layer_part: pd.Series) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as f:
        proc = {
            key: f[f"parts/process_parameters/{key}"][()]
            for key in f["parts/process_parameters"]
        }

    df_proc = (
        pd.DataFrame(proc)
        .rename_axis("part_id")
        .reset_index()
        .assign(part_id=lambda x: x.part_id + 1)  # HDF5 data are 0‑based
    )

    df_layer_params = (
        layer_part.reset_index()
        .merge(df_proc, on="part_id", how="left")
        .set_index("layer")
        .drop(columns="part_id")
    )
    logger.info("Process parameters broadcast → shape %s", df_layer_params.shape)
    return df_layer_params


# ─────────────────────────────────────────────────────────────
# 3.  Temporal 1‑D sensors (fast)
# ─────────────────────────────────────────────────────────────


def _load_temporal_sensors(h5_path: Path, *, n_layers: int) -> pd.DataFrame:
    """Read all temporal sensors truncated to *n_layers* (common timeline)."""
    with h5py.File(h5_path, "r") as f:
        sensors = {key: f[f"temporal/{key}"][:n_layers] for key in f["temporal"]}

    df = pd.DataFrame(sensors, index=pd.RangeIndex(n_layers, name="layer"))
    logger.info("Temporal sensors → shape %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# 4.  Optional per‑class segmentation pixel counts
# ─────────────────────────────────────────────────────────────


def _load_segmentation_counts(
    h5_path: Path,
    *,
    classes: Iterable[int] | None = None,
    n_layers: int,
) -> pd.DataFrame:
    """Return a frame of pixel counts for each requested class ID."""
    if classes is None:
        return pd.DataFrame(index=pd.RangeIndex(n_layers, name="layer"))

    with h5py.File(h5_path, "r") as f:
        class_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["slices/segmentation_results/class_names"][:]
        ]

        counts: dict[str, list[int]] = {}
        for cid in classes:
            cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            ds = f[f"slices/segmentation_results/{cid}"][0:n_layers]
            counts[f"px_{cname}"] = [
                int(layer.sum()) for layer in tqdm(ds, leave=False)
            ]

    df_seg = pd.DataFrame(counts, index=pd.RangeIndex(n_layers, name="layer"))
    logger.info("Segmentation counts → shape %s", df_seg.shape)
    return df_seg


# ─────────────────────────────────────────────────────────────
# 5.  Public builder
# ─────────────────────────────────────────────────────────────


def build_master_dataframe(
    h5_path: Path,
    *,
    out_csv: Path,
    workers: int | None = None,
) -> pd.DataFrame:
    """Create the master dataframe and write it to *out_csv*."""

    h5_path = h5_path.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    logger.info("Building master dataframe from %s", h5_path)

    # 5.1 modal part-ID → broadcast params
    layer_part = _layer_to_part_map(h5_path, workers=workers)
    df_layer_params = _broadcast_process_params(h5_path, layer_part)

    # 5.2 base features + targets (layer_dataframe)
    with h5py.File(h5_path, "r") as f:
        df_feat = layer_dataframe(f).set_index("layer").sort_index()
        n_common = len(df_feat)
        # add total powder‑bed pixels (Powder + Printed) limited to n_common
        powder = count_pixels_stream(f["slices/segmentation_results/0"], upto=n_common)
        printed = count_pixels_stream(f["slices/segmentation_results/1"], upto=n_common)
        df_feat["total_px"] = powder + printed
    logger.info("layer_dataframe (+total_px) → shape %s", df_feat.shape)

    # 5.3 temporal (truncated) & optional seg counts
    df_temporal = _load_temporal_sensors(h5_path, n_layers=n_common)
    df_seg = _load_segmentation_counts(
        h5_path,
        classes=[],
        n_layers=n_common,  # empty list → skip heavy counts
    )

    # 5.4 join all on common layer index
    df = (
        df_feat.join(df_temporal, how="left")
        .join(df_seg, how="left")
        .join(df_layer_params, how="left")
        .sort_index()
    )

    # move targets to end
    targ_cols = ["spatter_px", "streak_px"]
    other_cols = [c for c in df.columns if c not in targ_cols]
    df = df[other_cols + targ_cols]

    # 5.5 save to CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index_label="layer")
    logger.info("Saved master dataframe → %s (shape %s)", out_csv, df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# 6.  CLI entry‑point
# ─────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build layer‑indexed all feature CSV from Concept‑Laser HDF5 file."
    )
    p.add_argument("--h5", type=Path, required=True, help="Input HDF5 file")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("../data/build2_features.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: CPU count)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_master_dataframe(args.h5, out_csv=args.out, workers=args.workers)


if __name__ == "__main__":
    main()
