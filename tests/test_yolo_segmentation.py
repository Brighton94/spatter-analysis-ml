# tests/test_yolo_segmentation.py

import h5py
import numpy as np
import pytest
from src.utils.yolo_segment import (
    compute_anomaly_area,
    extract_anomaly_mask,
    load_hdf5_slice,
)


@pytest.fixture
def dummy_image_shape():
    return (100, 100)


@pytest.fixture
def dummy_results(monkeypatch):
    class DummyMask:
        def __init__(self, data, cls):
            self.data = data
            self.cls = cls

    class DummyBox:
        def __init__(self, cls_list):
            self.cls = np.array(cls_list)

    class DummyResult:
        def __init__(self):
            # two masks: one class 2, one class 3
            self.boxes = DummyBox([2, 3])
            self.masks = type(
                "M",
                (),
                {
                    "data": np.stack(
                        [
                            np.pad(
                                np.ones((50, 50)), ((25, 25), (25, 25))
                            ),  # mask at center
                            np.zeros((100, 100)),
                        ]
                    )
                },
            )

    return [DummyResult()]


def test_extract_anomaly_mask_includes_correct_classes(
    dummy_image_shape, dummy_results
):
    mask = extract_anomaly_mask(dummy_results, [2], dummy_image_shape)
    # should have 50x50 ones in center
    assert mask.dtype == np.uint8
    assert mask.sum() == 50 * 50


def test_compute_anomaly_area(dummy_image_shape):
    mask = np.zeros(dummy_image_shape, dtype=np.uint8)
    mask[0, :10] = 1
    area = compute_anomaly_area(mask, pixel_size_mm2=0.5)
    assert area == pytest.approx(10 * 0.5)


def test_load_hdf5_slice(tmp_path):
    # create dummy HDF5
    p = tmp_path / "test.h5"
    with h5py.File(p, "w") as f:
        grp = f.create_group("foo")
        grp.create_dataset("0", data=np.arange(16).reshape(4, 4).astype(np.uint8))
    arr = load_hdf5_slice(str(p), 0, "foo")
    assert arr.shape == (4, 4)
    assert arr.dtype == np.uint8
