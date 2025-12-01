"""Provide a simple HDF5 viewer for the Peregrine dataset (https://doi.ccs.ornl.gov/dataset/c0247625-951c-5616-a2e3-03803e848896).

- It allows users to visualize different datasets within the HDF5 file.
- It provides a GUI with options to select groups and layers.
- The data is displayed using matplotlib.
"""

import contextlib
import tkinter as tk

import h5py
import matplotlib
import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, BOTTOM, LEFT, RIGHT, YES, X

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, collections, colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.config import get_dataset_path

################################################################################
# Configuration
################################################################################
path_file = get_dataset_path("tcr_phase1_build2")
if path_file is None:
    raise FileNotFoundError(
        "Dataset not found. Please check the configuration and data directory."
    )

################################################################################
# We'll parse out the class names for segmentation if present
################################################################################
class_name_dict = {}
with h5py.File(path_file, "r") as f:
    seg_group = f.get("slices/segmentation_results", None)
    if seg_group is not None and "class_names" in seg_group.attrs:
        raw = seg_group.attrs["class_names"]
        splitted = [x.strip() for x in raw.split(",")]
        for i, name in enumerate(splitted):
            class_name_dict[i] = name


################################################################################
# 1) Collect only datasets from these top-level groups
################################################################################
def collect_datasets_grouped() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "slices/segmentation_results": [],
        "scans": [],
        "temporal": [],
        "reference_images": [],
        "micrographs": [],
    }
    with h5py.File(path_file, "r") as f:

        def visitor(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                parts = name.split("/")
                top2 = "/".join(parts[:2])
                top1 = parts[0]
                if top1 in groups:
                    groups[top1].append(name)
                elif top2 in groups:
                    groups[top2].append(name)

        f.visititems(visitor)

    for _k, datasets in groups.items():
        datasets.sort()
    return groups


################################################################################
# 2) Display logic
################################################################################
def display_dataset(ds_name: str, layer_idx: int, fig: plt.Figure) -> None:
    """Display the dataset in the given figure."""
    fig.clear()
    ax = fig.add_subplot(111)

    with h5py.File(path_file, "r") as build:
        if ds_name not in build:
            _display_missing_dataset(ax, ds_name, fig)
            return

        dset = build[ds_name]
        parts = ds_name.split("/")

        if (
            len(parts) >= 2
            and parts[0] == "slices"
            and parts[1] == "segmentation_results"
        ):
            _display_segmentation_results(ax, dset, ds_name, layer_idx)
        elif parts[0] == "scans":
            _display_scans(ax, dset, ds_name)
        elif parts[0] == "temporal":
            _display_temporal(ax, dset, ds_name)
        elif parts[0] in ["reference_images", "micrographs"]:
            _display_images(ax, dset, ds_name)
        else:
            _display_unhandled(ax, ds_name)

    fig.canvas.draw()


def _display_missing_dataset(ax: plt.Axes, ds_name: str, fig: plt.Figure) -> None:
    ax.text(
        0.5,
        0.5,
        f"Missing dataset:\n{ds_name}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title("Missing dataset")
    fig.canvas.draw()


def _display_segmentation_results(
    ax: plt.Axes, dset: h5py.Dataset, ds_name: str, layer_idx: int
) -> None:
    if layer_idx < dset.shape[0]:
        arr2d = dset[layer_idx, ...]
        seg_idx: int | None = None
        with contextlib.suppress(ValueError):
            seg_idx = int(ds_name.split("/")[-1])

        seg_label = class_name_dict.get(seg_idx or 0, f"Class {ds_name.split('/')[-1]}")
        ax.imshow(arr2d, cmap="jet", interpolation="none")
        ax.set_title(f"{ds_name}\nLayer={layer_idx} ({seg_label})")
    else:
        ax.text(
            0.5,
            0.5,
            f"layer={layer_idx} out of range\nshape={dset.shape}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(ds_name)


def _display_scans(ax: plt.Axes, dset: h5py.Dataset, ds_name: str) -> None:
    data = dset[...]
    x = data[:, 0:2]
    y = data[:, 2:4]
    t = data[:, 4]
    colorizer = cm.ScalarMappable(norm=colors.Normalize(t.min(), t.max()), cmap="jet")
    line_collection = collections.LineCollection(
        np.stack([x, y], axis=-1), colors=colorizer.to_rgba(t)
    )
    ax.add_collection(line_collection)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_title(ds_name)


def _display_temporal(ax: plt.Axes, dset: h5py.Dataset, ds_name: str) -> None:
    data = dset[...]
    ax.scatter(np.arange(len(data)), data, s=5)
    ax.set_title(ds_name)
    ax.set_xlabel("Layer index")
    units: str | bytes = dset.attrs.get("units", "")
    if isinstance(units, bytes):
        units = units.decode()
    ax.set_ylabel(units if units else "Value")


def _display_images(ax: plt.Axes, dset: h5py.Dataset, ds_name: str) -> None:
    arr = dset[...]
    ax.imshow(arr, interpolation="none")
    ax.set_title(ds_name)


def _display_unhandled(ax: plt.Axes, ds_name: str) -> None:
    ax.text(
        0.5,
        0.5,
        f"Unhandled: {ds_name}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title(ds_name)


################################################################################
# 3) Main GUI
################################################################################
def main() -> None:  # noqa: C901, PLR0915
    groups = collect_datasets_grouped()
    group_names = list(groups.keys())

    style = ttk.Style(theme="cosmo")
    app = style.master
    app.title("HDF5 Viewer - Changing Status Done/Loading")

    current_index: list[int] = [0]
    fig = plt.Figure(figsize=(7, 5))
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=BOTH, expand=YES)

    # Status label
    lbl_status = ttk.Label(app, text="", padding=5, anchor="center")
    lbl_status.pack(side=BOTTOM, fill=X)

    # Top controls
    top_frame = ttk.Frame(app, padding=5)
    top_frame.pack(fill=X)

    ttk.Label(top_frame, text="Group: ").pack(side=LEFT, padx=5)
    current_group_var = tk.StringVar(value=group_names[0])
    combo = ttk.Combobox(
        top_frame,
        textvariable=current_group_var,
        values=group_names,
        width=25,
        style="TCombobox",
    )
    combo.pack(side=LEFT, padx=5)

    ttk.Label(top_frame, text="Layer:").pack(side=LEFT, padx=5)
    layer_var = tk.IntVar(value=50)
    layer_spin = ttk.Spinbox(
        top_frame, from_=0, to=9999, textvariable=layer_var, width=6
    )  # noqa: E501
    layer_spin.pack(side=LEFT, padx=5)

    def on_go_layer() -> None:
        group = current_group_var.get()
        dsets = groups[group]
        layer_chosen = layer_var.get()
        if group == "scans":
            target = f"scans/{layer_chosen}"
            if target in dsets:
                current_index[0] = dsets.index(target)
                update_display()
            else:
                lbl_status.config(text=f"'{target}' not found in 'scans' group.")
        else:
            update_display()

    ttk.Button(top_frame, text="Go Layer", command=on_go_layer).pack(side=LEFT, padx=5)

    lbl_dataset = ttk.Label(app, text="", padding=5)
    lbl_dataset.pack(side=BOTTOM, fill=X)

    # Buttons for prev and next
    btn_frame = ttk.Frame(app, padding=5)
    btn_frame.pack(fill=X)

    def update_display() -> None:
        lbl_status.config(text="Loading data...", foreground="orange")
        lbl_status.update_idletasks()

        group = current_group_var.get()
        dsets = groups[group]
        if not dsets:
            lbl_dataset.config(text="No datasets in this group!")
            fig.clear()
            fig.canvas.draw()
            lbl_status.config(text="Done.", foreground="green")
            return

        ds_name = dsets[current_index[0]]
        lbl_dataset.config(text=f"{ds_name}  [{current_index[0] + 1}/{len(dsets)}]")

        display_dataset(ds_name, layer_var.get(), fig)
        lbl_status.config(text="Done.", foreground="green")

    def on_combo_changed(event: tk.Event | None = None) -> None:
        current_index[0] = 0
        update_display()

    combo.bind("<<ComboboxSelected>>", on_combo_changed)

    def on_prev() -> None:
        group = current_group_var.get()
        dsets = groups[group]
        if not dsets:
            return
        if current_index[0] > 0:
            current_index[0] -= 1
        else:
            current_index[0] = len(dsets) - 1
        update_display()

    def on_next() -> None:
        group = current_group_var.get()
        dsets = groups[group]
        if not dsets:
            return
        if current_index[0] < len(dsets) - 1:
            current_index[0] += 1
        else:
            current_index[0] = 0
        update_display()

    ttk.Button(btn_frame, text="Previous", command=on_prev).pack(side=LEFT, padx=5)
    ttk.Button(btn_frame, text="Next", command=on_next).pack(side=RIGHT, padx=5)

    update_display()
    app.mainloop()


if __name__ == "__main__":
    main()
