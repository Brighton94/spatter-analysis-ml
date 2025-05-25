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
from tqdm.auto import tqdm

from src.utils.feature_engineering import count_pixels_stream, layer_dataframe

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def _modal_part_id(idx: int, h5: str) -> tuple[int, int]:
    with h5py.File(h5, "r") as f:
        layer = f["slices/part_ids"][idx]
    nz = layer[layer > 0]
    return idx, int(np.bincount(nz).argmax()) if nz.size else 0


def _layer_to_part_map(h5: Path, *, workers: int | None) -> pd.Series:
    with h5py.File(h5, "r") as f:
        n_layers = f["slices/part_ids"].shape[0]

    workers = workers or (os.cpu_count() or 4)
    logger.info("Computing modal part-ID with %d worker(s) …", workers)

    part_ids = [0] * n_layers
    tic = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = (exe.submit(_modal_part_id, i, str(h5)) for i in range(n_layers))
        for fut in tqdm(
            as_completed(futs), total=n_layers, desc="modal part-ID", unit="layer"
        ):
            idx, pid = fut.result()
            part_ids[idx] = pid
    logger.info("…done in %.1f min", (time.perf_counter() - tic) / 60)
    return pd.Series(part_ids, name="part_id", dtype="int32").rename_axis("layer")


def _broadcast_process_params(h5: Path, layer_part: pd.Series) -> pd.DataFrame:
    with h5py.File(h5, "r") as f:
        keys = list(f["parts/process_parameters"].keys())
        proc = {
            k: f[f"parts/process_parameters/{k}"][()]
            for k in tqdm(keys, desc="reading process-parameters", unit="param")
        }
    df_proc = (
        pd.DataFrame(proc)
        .rename_axis("part_id")
        .reset_index()
        .assign(part_id=lambda x: x.part_id + 1)
    )  # 0-based → 1-based
    out = layer_part.reset_index().merge(df_proc, on="part_id", how="left")
    out = out.set_index("layer")  # keep part_id!
    logger.info("Process parameters broadcast → %s", out.shape)
    return out


def _load_temporal_sensors(h5: Path, *, n_layers: int) -> pd.DataFrame:
    with h5py.File(h5, "r") as f:
        keys = list(f["temporal"].keys())
        sensors = {
            k: f[f"temporal/{k}"][:n_layers]
            for k in tqdm(keys, desc="reading temporal sensors", unit="sensor")
        }
    df = pd.DataFrame(sensors, index=pd.RangeIndex(n_layers, name="layer"))
    logger.info("Temporal sensors → %s", df.shape)
    return df


# ------------------------------------------------------------------
# Per-class segmentation pixel counts  (streaming, memory-safe)
# ------------------------------------------------------------------
def _load_seg_counts(
    h5: Path, *, classes: Iterable[int], n_layers: int
) -> pd.DataFrame:
    """Return a DataFrame with pixel counts for every requested class."""

    if not classes:  # empty list → fast skip
        return pd.DataFrame(index=pd.RangeIndex(n_layers, name="layer"))

    with h5py.File(h5, "r") as f:
        grp = f["slices/segmentation_results"]

        # optional class-name lookup
        if "class_names" in grp:
            raw = grp["class_names"][:]
            names = [
                n.decode() if isinstance(n, bytes | np.bytes_) else str(n) for n in raw
            ]
        else:
            names = [f"class_{i}" for i in range(len(grp))]

        counts = {}
        for cid in classes:
            cname = names[cid] if cid < len(names) else f"class_{cid}"
            ds = grp[str(cid)]  # h5py.Dataset
            counts[f"px_{cname}"] = count_pixels_stream(
                ds, block_layers=32, axes=(1, 2), upto=n_layers
            )

    df = pd.DataFrame(counts, index=pd.RangeIndex(n_layers, name="layer"))
    df["total_px"] = df.filter(regex=r"^px_").sum(axis=1)
    logger.info("Segmentation counts → %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# builder
# ─────────────────────────────────────────────────────────────
def build_master_dataframe(
    h5: Path, *, out_csv: Path, workers: int | None = None
) -> pd.DataFrame:
    h5, out_csv = Path(h5).resolve(), Path(out_csv).resolve()
    logger.info("Building master dataframe from %s", h5)

    layer_part = _layer_to_part_map(h5, workers=workers)
    df_layer_params = _broadcast_process_params(h5, layer_part)

    with h5py.File(h5, "r") as f:
        df_feat = layer_dataframe(f).set_index("layer").sort_index()
        n_layers = len(df_feat)

    df_temporal = _load_temporal_sensors(h5, n_layers=n_layers)
    df_seg = _load_seg_counts(h5, classes=[0, 1], n_layers=n_layers)

    # drop temporal columns that are already present in df_feat
    dup_cols_tmp = df_feat.columns.intersection(df_temporal.columns)
    df_temporal = df_temporal.drop(columns=dup_cols_tmp)

    df = (
        df_feat.join(df_seg, how="left")
        .join(df_temporal, how="left")
        .join(df_layer_params, how="left")
        .assign(part_id=layer_part)  # ensure part_id is present
        .sort_index()
    )

    # targets last
    tcols = ["spatter_px", "streak_px"]
    df = df[[c for c in df.columns if c not in tcols] + tcols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index_label="layer")
    logger.info("Saved → %s  (shape %s)", out_csv, df.shape)
    return df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate layer-level feature CSV")
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/build2_features.csv"))
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    a = _parse()
    build_master_dataframe(a.h5, out_csv=a.out, workers=a.workers)


if __name__ == "__main__":
    main()
