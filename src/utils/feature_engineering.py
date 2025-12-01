from __future__ import annotations

from collections.abc import Sequence

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def count_pixels_stream(
    ds: h5py.Dataset,
    *,
    block_layers: int = 32,
    axes: Sequence[int] = (1, 2),
    upto: int | None = None,
) -> np.ndarray:
    """Vectorised pixel count, streamed in blocks, *with* progress bar."""

    n_layers = ds.shape[0] if upto is None else min(ds.shape[0], upto)
    counts = np.empty(n_layers, dtype=np.int32)

    for start in tqdm(
        range(0, n_layers, block_layers),
        desc="count_pixels",
        unit="layer",
        leave=False,
    ):
        stop = min(start + block_layers, n_layers)
        slab = ds[start:stop]
        counts[start:stop] = np.count_nonzero(slab, axis=axes)

    return counts


def layer_dataframe(h5: h5py.File) -> pd.DataFrame:
    n_layers_masks = h5["slices/segmentation_results/8"].shape[0]
    n_layers_sensors = h5["temporal/gas_loop_oxygen"].shape[0]

    n = min(n_layers_masks, n_layers_sensors)

    df = pd.DataFrame(
        {
            "layer": np.arange(n),
            "gas_oxygen": h5["temporal/gas_loop_oxygen"][:n],
            "top_flow_rate": h5["temporal/top_flow_rate"][:n],
        }
    )

    # anomaly pixel counts
    df["spatter_px"] = count_pixels_stream(h5["slices/segmentation_results/8"], upto=n)
    df["streak_px"] = count_pixels_stream(h5["slices/segmentation_results/3"], upto=n)

    return df


__all__ = ["count_pixels_stream", "layer_dataframe"]
