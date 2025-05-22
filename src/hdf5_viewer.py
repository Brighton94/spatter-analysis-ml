"""HDF5 viewer for the Peregrine LPBF dataset."""

from __future__ import annotations

import tkinter as tk
from collections import defaultdict

import h5py
import matplotlib
import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, BOTTOM, LEFT, YES, X

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import collections, colormaps, colors  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from src.config import get_dataset_path  # noqa: E402

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
path_file = get_dataset_path("tcr_phase1_build2")
if path_file is None:
    raise FileNotFoundError(
        "Dataset not found. Check configuration and data directory."
    )

# Parse segmentation class names
class_name_dict: dict[int, str] = {}
with h5py.File(path_file, "r") as _f:
    seg_group = _f.get("slices/segmentation_results")
    if seg_group is not None and "class_names" in seg_group.attrs:
        raw = seg_group.attrs["class_names"]
        for idx, name in enumerate(x.strip() for x in raw.split(",")):
            class_name_dict[idx] = name

# Dataset discovery
EXCLUDE_KEYWORDS = {"parts", "samples"}
KEEP_TOP = {
    "slices/segmentation_results",
    "slices/part_ids",
    "slices/camera_data",
    "scans",
    "temporal",
    "reference_images",
    "micrographs",
}


def collect_datasets_grouped() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with h5py.File(path_file, "r") as f:

        def visitor(name: str, obj: h5py.HLObject) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if any(kw in name for kw in EXCLUDE_KEYWORDS):
                return
            top = name.split("/")[0]
            top2 = "/".join(name.split("/")[:2])
            key = top2 if top2 in KEEP_TOP else top
            if key in KEEP_TOP:
                groups[key].append(name)

        f.visititems(visitor)
    for lst in groups.values():
        lst.sort()
    return dict(groups)


# ---------------------------------------------------------------------
# Display routines
# ---------------------------------------------------------------------


def display_dataset(ds_name: str, layer_idx: int, fig: plt.Figure) -> None:  # noqa: C901
    fig.clear()
    ax = fig.add_subplot(111)

    with h5py.File(path_file, "r") as build:
        if ds_name not in build:
            ax.text(
                0.5,
                0.5,
                f"Missing dataset:\n{ds_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Missing dataset")
            fig.canvas.draw_idle()
            return

        dset = build[ds_name]
        parts = ds_name.split("/")

        if parts[:2] == ["slices", "segmentation_results"]:
            _show_segmentation(ax, dset, ds_name, layer_idx)
        elif ds_name == "slices/part_ids":
            _show_part_ids(ax, dset, layer_idx)
        elif parts[:2] == ["slices", "camera_data"]:
            _show_camera(ax, dset, ds_name, layer_idx)
        elif parts[0] == "scans":
            _show_scans(ax, dset, ds_name)
        elif parts[0] == "temporal":
            _show_temporal(ax, dset, ds_name)
        elif parts[0] in {"reference_images", "micrographs"}:
            _show_image(ax, dset, ds_name)
        else:
            ax.text(
                0.5,
                0.5,
                f"Unhandled: {ds_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(ds_name)

    fig.canvas.draw_idle()


def _layer_guard(ax: plt.Axes, name: str, layer: int, shape: tuple[int, ...]) -> bool:
    if layer >= shape[0]:
        ax.text(
            0.5,
            0.5,
            f"layer={layer} out of range\nshape={shape}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(name)
        return True
    return False


def _show_segmentation(ax: plt.Axes, dset: h5py.Dataset, name: str, layer: int) -> None:  # noqa: ANN001
    if _layer_guard(ax, name, layer, dset.shape):
        return
    arr = dset[layer]
    seg_id = int(name.split("/")[-1])
    label = class_name_dict.get(seg_id, f"Class {seg_id}")
    ax.imshow(arr, cmap="jet", interpolation="none")
    ax.set_title(f"{name}\nLayer {layer} · {label}")
    ax.axis("off")


def _show_part_ids(ax: plt.Axes, dset: h5py.Dataset, layer: int) -> None:  # noqa: ANN001
    if _layer_guard(ax, "slices/part_ids", layer, dset.shape):
        return
    arr = dset[layer].astype(np.int32)
    max_id = int(arr.max())
    cmap = colormaps["tab20"] if max_id <= 20 else colormaps["nipy_spectral"]
    im = ax.imshow(
        arr, cmap=cmap, interpolation="none", vmin=0, vmax=max_id if max_id else 1
    )
    ax.set_title(f"part_ids · Layer {layer} · max ID {max_id}")
    ax.axis("off")
    if max_id <= 20:
        fig = ax.figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _show_camera(ax: plt.Axes, dset: h5py.Dataset, name: str, layer: int) -> None:  # noqa: ANN001
    if _layer_guard(ax, name, layer, dset.shape):
        return
    img = dset[layer]
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", interpolation="none")
    else:
        ax.imshow(img, interpolation="none")
    ax.set_title(f"{name} · Layer {layer}")
    ax.axis("off")


def _show_scans(ax: plt.Axes, dset: h5py.Dataset, name: str) -> None:  # noqa: ANN001
    data = dset[...]
    x = data[:, 0:2]
    y = data[:, 2:4]
    t = data[:, 4]
    cmap = colormaps["jet"]
    norm = colors.Normalize(t.min(), t.max())
    lc = collections.LineCollection(np.stack([x, y], axis=-1), colors=cmap(norm(t)))
    ax.add_collection(lc)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_title(name)


def _show_temporal(ax: plt.Axes, dset: h5py.Dataset, name: str) -> None:  # noqa: ANN001
    data = dset[...]
    ax.scatter(range(len(data)), data, s=4)
    units = dset.attrs.get("units", "")
    if isinstance(units, bytes):
        units = units.decode()
    ax.set_xlabel("Layer index")
    ax.set_ylabel(units or "Value")
    ax.set_title(name)


def _show_image(ax: plt.Axes, dset: h5py.Dataset, name: str) -> None:  # noqa: ANN001
    ax.imshow(dset[...], interpolation="none")
    ax.set_title(name)
    ax.axis("off")


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------


def main() -> None:  # noqa: C901, PLR0915
    groups = collect_datasets_grouped()
    group_names = list(groups)

    style = ttk.Style(theme="cosmo")
    root = style.master
    root.title("Peregrine HDF5 viewer")

    fig = plt.Figure(figsize=(7, 5))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=BOTH, expand=YES)

    status = ttk.Label(root, text="", padding=4, anchor="center")
    status.pack(side=BOTTOM, fill=X)
    dataset_lbl = ttk.Label(root, text="", padding=4)
    dataset_lbl.pack(side=BOTTOM, fill=X)

    top = ttk.Frame(root, padding=4)
    top.pack(fill=X)

    ttk.Label(top, text="Group:").pack(side=LEFT, padx=4)
    group_var = tk.StringVar(value=group_names[0])
    cbo = ttk.Combobox(
        top, textvariable=group_var, values=group_names, width=28, state="readonly"
    )
    cbo.pack(side=LEFT, padx=4)

    ttk.Label(top, text="Layer:").pack(side=LEFT, padx=4)
    layer_var = tk.IntVar(value=0)
    spin = ttk.Spinbox(top, from_=0, to=999999, textvariable=layer_var, width=6)
    spin.pack(side=LEFT, padx=4)

    cur_idx = [0]

    def refresh() -> None:
        status.configure(text="Loading…", foreground="orange")
        status.update_idletasks()
        group = group_var.get()
        dsets = groups.get(group, [])
        if not dsets:
            dataset_lbl.configure(text="(no datasets)")
            fig.clear()
            fig.canvas.draw_idle()
            status.configure(text="Done", foreground="green")
            return
        ds = dsets[cur_idx[0] % len(dsets)]
        dataset_lbl.configure(text=f"{ds}  [{cur_idx[0] + 1}/{len(dsets)}]")
        display_dataset(ds, layer_var.get(), fig)
        status.configure(text="Done", foreground="green")

    def change_group(event: tk.Event | None = None) -> None:  # noqa: ANN001
        cur_idx[0] = 0
        refresh()

    cbo.bind("<<ComboboxSelected>>", change_group)

    def prev_ds() -> None:  # noqa: ANN001
        cur_idx[0] -= 1
        refresh()

    def next_ds() -> None:  # noqa: ANN001
        cur_idx[0] += 1
        refresh()

    ttk.Button(top, text="Prev", command=prev_ds).pack(side=LEFT, padx=4)
    ttk.Button(top, text="Next", command=next_ds).pack(side=LEFT, padx=4)
    ttk.Button(top, text="Go", command=refresh).pack(side=LEFT, padx=4)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
