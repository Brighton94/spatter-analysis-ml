from __future__ import annotations

import h5py
import numpy as np
import pandas as pd


def part_anomaly_fractions(
    h5: h5py.File,
    sp_id: int = 8,
    st_id: int = 3,
    layers_per_chunk: int = 8,
) -> pd.DataFrame:
    """Compute per-part anomaly fraction (spatter ∨ streak) without loading the full 3-D volume."""  # noqa: E501

    ds_part = h5["slices/part_ids"]
    ds_spat = h5[f"slices/segmentation_results/{sp_id}"]
    ds_streak = h5[f"slices/segmentation_results/{st_id}"]

    n_layers = ds_part.shape[0]
    max_pid = int(ds_part.attrs.get("max_part_id", 0))  # optional metadata
    if max_pid == 0:  # fallback: first cheap scan (no masks)
        max_pid = int(ds_part[:layers_per_chunk].max())

    total_px = np.zeros(max_pid + 1, dtype=np.int64)
    anomaly_px = np.zeros_like(total_px)

    for start in range(0, n_layers, layers_per_chunk):
        stop = min(start + layers_per_chunk, n_layers)

        part_blk = ds_part[start:stop]  # shape (L, H, W)  uint32
        sp_blk = ds_spat[start:stop]  # bool
        st_blk = ds_streak[start:stop]  # bool
        an_blk = np.logical_or(sp_blk, st_blk)  # bool

        # flatten so we can `bincount` on part IDs
        part_flat = part_blk.ravel()
        an_flat = an_blk.ravel().astype(np.int8)

        # ensure arrays large enough if we discover a higher ID mid-stream
        pid_max_blk = int(part_flat.max())
        if pid_max_blk >= total_px.size:
            grow_by = pid_max_blk - total_px.size + 1
            total_px = np.pad(total_px, (0, grow_by))
            anomaly_px = np.pad(anomaly_px, (0, grow_by))

        total_px += np.bincount(part_flat, minlength=total_px.size)
        anomaly_px += np.bincount(part_flat, weights=an_flat, minlength=total_px.size)

    # drop PID = 0 (background)
    part_idx = np.nonzero(total_px)[0]  # keeps only parts actually present
    part_idx = part_idx[part_idx != 0]

    frac = anomaly_px[part_idx] / total_px[part_idx]

    return pd.DataFrame(
        {
            "anomaly_frac": frac,
            "total_px": total_px[part_idx],
            "anomaly_px": anomaly_px[part_idx],
        },
        index=part_idx,
    ).sort_values("anomaly_frac")
