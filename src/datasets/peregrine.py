"""Lazy-loading Dataset for multiclass Peregrine L-PBF images + masks."""

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class PeregrineDataset(Dataset):
    """Dataset for Peregrine L-PBF images and masks.

    0 = background
    1 = streak
    2 = spatter
    """

    def __init__(self, h5_path, layers=None, size=512, augment=False):
        self.h5_path = h5_path  # path only; file opened per worker
        self.h5 = None
        self.layers = list(layers) if layers is not None else None

        # transform for images
        img_t = [T.Resize(size), T.ToTensor()]
        if augment:
            img_t.insert(1, T.RandomHorizontalFlip())
            img_t.insert(2, T.RandomVerticalFlip())
        self.img_tf = T.Compose(img_t)

        # resize-only transform for masks (nearest neighbour)
        self.mask_tf = lambda mask: torch.from_numpy(
            np.array(mask.resize((size, size), resample=Image.NEAREST))
        ).long()

    def _lazy_init(self):
        if self.h5 is None:
            self.h5 = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=256 * 1024**2,  # 256 MB chunk cache
                swmr=True,  # safe multi-process reads
            )
            self.imgs = self.h5["slices/camera_data/visible/0"]
            self.sp = self.h5["slices/segmentation_results/8"]  # spatter
            self.st = self.h5["slices/segmentation_results/3"]  # streak

    def __len__(self):
        self._lazy_init()
        return len(self.layers) if self.layers is not None else len(self.imgs)

    def __getitem__(self, idx):
        self._lazy_init()
        layer = self.layers[idx] if self.layers is not None else idx

        # load & convert image
        img_arr = self.imgs[layer]
        img = Image.fromarray(img_arr).convert("RGB")
        img_t = self.img_tf(img)  # [3, H, W]

        # load raw masks and build class map
        sp = self.sp[layer]
        st = self.st[layer]
        lbl = np.zeros_like(sp, dtype=np.uint8)  # background
        lbl[st > 0] = 1  # streak
        lbl[sp > 0] = 2  # spatter
        mask = Image.fromarray(lbl)  # PIL for resizing
        mask_t = self.mask_tf(mask)  # [H, W] int64

        return img_t, mask_t
