# Physics-Informed Machine Learning for Metal AM


**Status:** _Work-in-progress (WIP)_. This repo is can be used as a template for 'MLOps for Research.' The goal is to use physics-informed machine learning models and Vision Transformers (ViT, SegFormer) for segmentation.

## Setup Instructions

### Environment Variables

If you're using Linux and want to run GUI applications inside the container, you need to:

1. Run this command on your host machine before starting the container:
   ```bash
   xhost +local:
   ```

2. Verify your DISPLAY environment variable:
   ```bash
   echo $DISPLAY
   ```

### Configuration

#### Container Configuration

Modify `.devcontainer/devcontainer.json` according to your system:

1.  **Mounts**: Update the `mounts` array to point to your dataset location on the host machine. The `target` path is where the data will be accessible inside the container. For example:
    ```json
    "mounts": [
      "source=/path/to/your/data,target=/mnt/ssd,type=bind,consistency=cached"
    ],
    ```

#### Dataset Paths

The application expects the dataset files to be available within the container at the `target` path specified in the `mounts` section of `.devcontainer/devcontainer.json` (e.g., `/mnt/ssd`).

The `src/config.py` script will look for specific dataset files (like `tcr_phase1_build1.hdf5`) within this mounted directory. Ensure your host directory (the `source` in the mount configuration) contains the necessary HDF5 files.

### Training a Segmentation Model

To train a Hugging Face SegFormer (ViT) model on the available datasets, use the provided shell script:

```bash
./experiments/train.sh
```

This script will:
- Install all dependencies (if not already installed) using `uv` and `pyproject.toml`.
- Launch the training pipeline (`src/training/train_full`) for the available builds (e.g., `tcr_phase1_build1`, `tcr_phase1_build2`).
- Use the SegFormer model from Hugging Face Transformers for semantic segmentation.

Model checkpoints, logs, and experiment outputs will be saved in the appropriate directories (see `runs/` and `experiments/`).
