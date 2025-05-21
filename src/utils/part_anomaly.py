from __future__ import annotations

import math

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def part_anomaly_fractions(
    h5: h5py.File,
    *,
    sp_id: int = 8,
    st_id: int = 3,
    layers_per_chunk: int | None = None,
    desc: str = "Per-part anomaly",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute per-part anomaly fraction (spatter ∨ streak)."""

    ds_part = h5["slices/part_ids"]
    ds_spat = h5[f"slices/segmentation_results/{sp_id}"]
    ds_streak = h5[f"slices/segmentation_results/{st_id}"]

    if layers_per_chunk is None:
        layers_per_chunk = ds_part.chunks[0] if ds_part.chunks else 32

    n_layers = ds_part.shape[0]
    n_iters = math.ceil(n_layers / layers_per_chunk)
    max_pid_guess = int(ds_part.attrs.get("max_part_id", 0)) or int(
        ds_part[:layers_per_chunk].max()
    )

    total_px = np.zeros(max_pid_guess + 1, dtype=np.int64)
    anomaly_px = np.zeros_like(total_px)

    # Pre-allocate reusable read buffers (shape = (layers_per_chunk, H, W))
    part_buf = np.empty((layers_per_chunk, *ds_part.shape[1:]), ds_part.dtype)
    spat_buf = np.empty_like(part_buf, dtype=bool)
    streak_buf = np.empty_like(part_buf, dtype=bool)

    rng = range(0, n_layers, layers_per_chunk)
    if show_progress:
        rng = tqdm(rng, total=n_iters, desc=desc, unit="blk")

    for start in rng:
        stop = min(start + layers_per_chunk, n_layers)
        n_this = stop - start
        dest_slice = slice(0, n_this)

        # single read per dataset → into pre-allocated buffers
        ds_part.read_direct(
            part_buf, source_sel=slice(start, stop), dest_sel=dest_slice
        )
        ds_spat.read_direct(
            spat_buf, source_sel=slice(start, stop), dest_sel=dest_slice
        )
        ds_streak.read_direct(
            streak_buf, source_sel=slice(start, stop), dest_sel=dest_slice
        )

        part_flat = part_buf[:n_this].reshape(-1)  # view, no copy
        mask = np.logical_or(spat_buf[:n_this], streak_buf[:n_this]).reshape(-1)

        pid_max_blk = int(part_flat.max())
        if pid_max_blk >= total_px.size:  # grow arrays on-the-fly
            grow = pid_max_blk - total_px.size + 1
            total_px = np.pad(total_px, (0, grow))
            anomaly_px = np.pad(anomaly_px, (0, grow))

        # accumulate counts in-place
        np.add.at(total_px, part_flat, 1)
        np.add.at(anomaly_px, part_flat[mask], 1)

    # Build dataframe (drop background PID 0 and never-seen IDs)
    valid = total_px != 0
    valid[0] = False
    idx = np.flatnonzero(valid)

    return pd.DataFrame(
        {
            "anomaly_frac": anomaly_px[idx] / total_px[idx],
            "total_px": total_px[idx],
            "anomaly_px": anomaly_px[idx],
        },
        index=idx,
    ).sort_values("anomaly_frac")
