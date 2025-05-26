"""Quickly extract five example images and masks for each anomaly class (0–11)."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def find_first_n_layers_with_anomaly(
    mask_ds: h5py.Dataset, n: int = 5, chunk_size: int = 100
) -> list[int]:
    """Scan mask_ds in chunks of layers to find up to n layer indices.

    where the anomaly is present, without loading the entire volume.
    """

    total_layers = mask_ds.shape[0]
    found: list[int] = []
    for start in range(0, total_layers, chunk_size):
        stop = min(start + chunk_size, total_layers)
        block = mask_ds[start:stop]  # shape = (chunk_size, H, W)
        presence = np.any(block, axis=(1, 2))  # one bool per layer in block
        for i, has in enumerate(presence):
            if has:
                idx = start + i
                found.append(idx)
                logger.debug(f"Found anomaly in layer {idx}")
                if len(found) >= n:
                    return found
    logger.info(f"Only found {len(found)} layers with anomaly (requested {n})")
    return found


def save_image_and_mask(
    img_ds: h5py.Dataset,
    mask_ds: h5py.Dataset,
    layer_idx: int,
    class_id: int,
    out_dir: Path,
) -> None:
    """Load one image + mask at layer_idx and save them as .npy files."""

    img = img_ds[layer_idx]  # shape = (H, W, ...) depending on your data
    mask = mask_ds[layer_idx]  # shape = (H, W)

    img_path = out_dir / f"class_{class_id:02d}_layer_{layer_idx:04d}_img.npy"
    mask_path = out_dir / f"class_{class_id:02d}_layer_{layer_idx:04d}_mask.npy"

    np.save(img_path, img)
    np.save(mask_path, mask)
    logger.debug(f"Saved image → {img_path}, mask → {mask_path}")


def main(h5_path: Path, output_root: Path) -> None:
    """Extract up to five example images and masks for each anomaly class (0–11).

    1. Find up to five layers containing that anomaly (chunked).
    2. Concurrently extract and save image + mask pairs,
       using a small thread pool to limit memory spikes.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        img_ds = f["slices/camera_data/visible/0"]
        seg_base = f["slices/segmentation_results"]

        # Prepare tasks: (class_id, [layer indices...])
        tasks: list[tuple[int, list[int]]] = []
        for class_id in range(12):
            mask_ds = seg_base[str(class_id)]
            idxs = find_first_n_layers_with_anomaly(mask_ds, n=5, chunk_size=100)
            tasks.append((class_id, idxs))

        # Concurrently load & save, but cap workers to avoid too many simultaneous reads
        with ThreadPoolExecutor(max_workers=12) as exe:
            futures = []
            for class_id, idxs in tasks:
                class_out = output_root / f"class_{class_id:02d}"
                class_out.mkdir(exist_ok=True)
                mask_ds = seg_base[str(class_id)]
                for layer_idx in idxs:
                    futures.append(
                        exe.submit(
                            save_image_and_mask,
                            img_ds,
                            mask_ds,
                            layer_idx,
                            class_id,
                            class_out,
                        )
                    )
            # Wait for all to complete
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception:
                    logger.exception("Error while saving one of the image/mask pairs")

    logger.info("Extraction complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} "
            "/piml-in-metal-am/data/2021-04-16 TCR Phase 1 Build 2.hdf5 "
            "/piml-in-metal-am/data/images/concept-laser-anomalies"
        )
        sys.exit(1)

    h5_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    main(h5_file, out_dir)
