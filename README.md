# UniCalib: Targetless LiDAR–Camera Calibration via Probabilistic Flow on Unified Depth Representations

Official repository for the paper:

> **UniCalib: Targetless LiDAR–Camera Calibration via Probabilistic Flow on Unified Depth Representations**  
> Shu Han, Xubo Zhu, Ji Wu, Ximeng Cai, Wen Yang, Huai Yu, and Gui-Song Xia  
> **WACV 2026**

[[Paper]](https://openaccess.thecvf.com/content/WACV2026/papers/Han_UniCalib_Targetless_LiDAR-camera_Calibration_via_Probabilistic_Flow_on_Unified_Depth_WACV_2026_paper.pdf)

---

## Overview

**UniCalib** introduces a probabilistic framework for *targetless LiDAR–camera calibration* by unifying both modalities into dense depth representations.  
We model the calibration problem as **probabilistic flow estimation** in the unified depth space, incorporating flow uncertainty and a perceptually weighted sparse flow loss to achieve robust cross-sensor alignment.

### Key Highlights

- **Probabilistic depth flow** reframing 2D–3D correspondence for targetless calibration
- **Unified depth representation** bridging LiDAR and camera via a shared encoder
- **Reliability-aware modeling** with the novel **PWSF loss** for robust optimization
- **Strong generalization** with accurate results across diverse datasets

---

## Installation

### Dependencies

```bash
# Create virtual environment (recommended)
conda create -n unicalib python=3.10
conda activate unicalib

# Install PyTorch (select based on your CUDA version)
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Visibility Module

This project requires a custom CUDA extension for depth image generation and occlusion handling:

```bash
cd visibility
pip install .
```

---

## Data Preparation

### Step 1: Generate Monocular Depth Estimation

Before training or testing, you need to generate depth estimation results from RGB images using a monocular depth estimation model (e.g., [MoGe](https://github.com/microsoft/MoGe), [Depth Anything](https://github.com/LiheYoung/Depth-Anything)).

For each RGB image, generate the corresponding depth map and save it as `depth_normalized.npy` in the appropriate folder structure (see dataset structures below).


The depth output should be normalized and saved as `.npy` files with the same filename as the corresponding RGB image.

### Step 2: Organize Dataset

#### Supported Datasets

| Dataset | Config Name | Description |
|---------|-------------|-------------|
| KITTI Odometry | `DatasetKITTIOdo` | Standard KITTI odometry dataset |
| KITTI Raw | `DatasetKITTIRaw` | KITTI raw dataset |
| KITTI-360 | `DatasetKITTI360` | KITTI-360 dataset |
| Waymo | `DatasetWaymo` | Waymo Open Dataset |
| Custom | `DatasetCustomTest` | User-defined dataset (test only) |

---

### KITTI Odometry Dataset Structure

```
Data/KITTI_all/sequences/
├── 00/
│   ├── velodyne/                    # Point cloud files (.bin)
│   │   ├── 000000.bin
│   │   └── ...
│   ├── image_2/                     # Original RGB images
│   │   ├── 000000.png
│   │   └── ...
│   ├── <depth_folder>/              # Monocular depth estimation results
│   │   ├── 000000/
│   │   │   ├── depth_normalized.npy # Normalized depth map
│   │   │   └── image.jpg            # RGB image (optional, for visualization)
│   │   └── ...
│   └── calib.txt                    # Calibration file
├── 01/
└── ...
```

---

### KITTI Raw Dataset Structure

```
Data/KITTI_raw/
├── 2011_09_26/
│   ├── 2011_09_26_drive_0001_sync/
│   │   ├── velodyne_points/
│   │   │   └── data/                # Point cloud files (.bin)
│   │   │       ├── 0000000000.bin
│   │   │       └── ...
│   │   └── image_02/
│   │       ├── data/                # Original RGB images
│   │       │   ├── 0000000000.png
│   │       │   └── ...
│   │       └── <depth_folder>/      # Monocular depth estimation results
│   │           ├── 0000000000/
│   │           │   └── depth_normalized.npy
│   │           └── ...
│   ├── 2011_09_26_drive_0002_sync/
│   └── ...
├── 2011_09_30/
└── ...
```

---

### KITTI-360 Dataset Structure

```
Data/KITTI-360/
├── 2013_05_28_drive_0000_sync/
│   ├── velodyne_points/
│   │   └── data/                    # Point cloud files (.bin)
│   │       ├── 0000000000.bin
│   │       └── ...
│   ├── image_00/
│   │   └── data_rect/               # Original RGB images
│   │       ├── 0000000000.png
│   │       └── ...
│   └── <depth_folder>/              # Monocular depth estimation results
│       ├── 0000000000/
│       │   └── depth_normalized.npy
│       └── ...
├── 2013_05_28_drive_0002_sync/
└── ...
```

---

### Waymo Dataset Structure

```
Data/Waymo/
├── segment-xxx/
│   ├── lidar/                       # Point cloud files (.h5)
│   │   ├── 000000.h5
│   │   └── ...
│   ├── images/                      # Original RGB images
│   │   ├── 000000.jpg
│   │   └── ...
│   └── <depth_folder>/              # Monocular depth estimation results
│       ├── 000000/
│       │   └── depth_normalized.npy
│       └── ...
├── segment-yyy/
└── ...
```

---

### Custom Dataset Structure

```
/your/data/root/
├── images/                          # RGB images (dataset scans this folder)
│   ├── 000000.jpg
│   └── ...
├── depth_imgs/                      # Monocular depth estimation results
│   ├── 000000/
│   │   └── depth_normalized_log.npy
│   └── ...
├── lidar/                           # Point cloud files (.bin or .npy)
│   ├── 000000.bin
│   └── ...
├── extrinsics/                      # Camera poses (optional, 4x4 matrix)
│   ├── 000000.txt
│   └── ...
└── lidar_pose/                      # LiDAR poses (optional, 4x4 matrix)
    ├── 000000.txt
    └── ...
```

**File Naming Rules:**
- RGB images, depth images, and point clouds must have matching filenames (without extension)
- Samples without corresponding depth images or point clouds will be automatically skipped

### Supported File Formats

| Type | Formats |
|------|---------|
| Point Cloud | `.bin` (KITTI format), `.npy`, `.h5` (Waymo) |
| Depth Image | `.npy` |
| RGB Image | `.jpg`, `.png`, `.jpeg` |

---

## Configuration

Configuration files use TOML format and are located in the `cfg/` directory.

### Training Config Example (`cfg/train.toml`)

```toml
title = "UniCalib-Train"
gpus = [0]
mode = "train"

[experiment]
seed = 3407

[dataset]
name = "DatasetKITTIOdo"
root_folder = 'Data/KITTI_all/sequences'
pcl_folder = 'velodyne'
imgs_folder = 'depth_moge'           # Folder containing depth estimation results
depth_imgs_folder = 'depth_moge'
train_sequences = ["01","02","03","04","05","06","07","08","09","10"]
test_sequence = '00'
max_r = 5.0                          # Max rotation perturbation (degrees)
max_t = 0.1                          # Max translation perturbation (meters)
batch_size = 6
num_workers = 12
w_scale = 1                          # Image width scale
h_scale = 1                          # Image height scale

[trainer]
max_epoch = 100
grad_acc_steps = 1
val_steps = 1

[optimizer]
type = "AdamW"
lr = 1e-4
weight_decay = 1e-5

[scheduler]
type = "OneCycleLR"

[model]
name = "RAFT"
iters = 4
dim = 128
radius = 4
block_dims = [64, 128, 256]
pretrain = "resnet34"
```

### Test Config Example (`cfg/test.toml`)

```toml
title = "UniCalib-Test"
gpus = [0]
mode = "test"

[experiment]
seed = 3407
if_render = true                     # Save visualization results

[dataset]
name = "DatasetKITTIOdo"
root_folder = 'Data/KITTI_all/sequences'
test_sequence = '00'
# ... other parameters same as training config
```

### Custom Dataset Config (`cfg/infer.toml`)

```toml
title = "infer"
gpus = [0]
mode = "test"

[experiment]
seed = 3407
if_render = true

[dataset]
name = "DatasetCustomTest"
root_folder = '/path/to/your/data'
pcl_folder = 'lidar'
imgs_folder = 'images'
depth_imgs_folder = 'depth_imgs'
extrinsics_folder = 'extrinsics'
lidar_poses_folder = 'lidar_pose'

max_r = 0.0                          # Set to 0 for no perturbation
max_t = 0.0
batch_size = 1
w_scale = 0.5
h_scale = 0.5

# LiDAR coordinate transform parameters
rotate_lidar_deg = 0.0               # Rotation around Z-axis (degrees)
scale_lidar = 1.0                    # Point cloud scale factor
translate_lidar_z = 0.0              # Translation along Z-axis (meters)

[model]
name = "RAFT"
# ... model parameters
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `root_folder` | str | Data root directory |
| `pcl_folder` | str | Point cloud folder name |
| `imgs_folder` | str | Depth estimation folder name |
| `depth_imgs_folder` | str | Depth estimation folder name |
| `max_r` | float | Max rotation perturbation (degrees) |
| `max_t` | float | Max translation perturbation (meters) |
| `w_scale` / `h_scale` | float | Image scale ratio |
| `adjust_coordinate_order` | list | Coordinate axis order adjustment |
| `if_render` | bool | Save visualization results |

---

## Usage

### Training

```bash
python train.py --cfg cfg/train.toml --log_steps 5

# Resume from checkpoint
python train.py --cfg cfg/train.toml --checkpoint path/to/checkpoint.pth --resume

# Multi-GPU training
torchrun --nproc_per_node=2 train.py --cfg cfg/train.toml
```

### Testing

```bash
python test.py --cfg cfg/test.toml --checkpoint checkpoints/model.pth
```

### Custom Data Inference

```bash
python test.py --cfg cfg/infer.toml --checkpoint checkpoints/model.pth
```

---

## Output

After testing, results are saved in the `output/` directory:

```
output/
├── iter_0/
│   ├── 01_rgb_original.png          # Original RGB image
│   ├── 02_mono_depth_estimated.png  # Monocular depth estimation
│   ├── 03_gt_lidar_sparse.png       # Sparse LiDAR depth (GT)
│   ├── 04_gt_lidar_dense.png        # Dense LiDAR depth (GT)
│   ├── 06_perturbed_lidar_dense.png # Perturbed LiDAR depth
│   ├── 10_dense_lidar_pre.png       # Predicted LiDAR depth
│   └── 12_flow_pred_transed.png     # Predicted optical flow
├── iter_1/
└── result_dict.csv                  # Evaluation metrics
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `t_Mean` | Mean translation error |
| `tX/tY/tZ` | Translation error per axis |
| `R_Mean` | Mean rotation error |
| `RX/RY/RZ` | Rotation error per axis |
| `EPE` | End Point Error |
| `F1` | F1 score |

---

## Project Structure

```
UniCalib/
├── cfg/                    # Configuration files
├── checkpoints/            # Model weights
├── visibility/             # CUDA extension for depth processing
├── core/
│   ├── model.py           # RAFT model
│   ├── layers.py          # Network layers
│   ├── trainer.py         # Trainer
│   ├── tester.py          # Tester
│   ├── evaluation.py      # Evaluation functions
│   ├── constant/          # Constants
│   ├── utils/             # Utilities
│   └── deepL/
│       ├── datasets/      # Dataset classes
│       ├── engine/        # Training/testing engine
│       ├── evaluation/    # Evaluation module
│       ├── model/         # Model registry
│       └── tensor_ops/    # Tensor operations
├── train.py               # Training entry
├── test.py                # Testing entry
└── requirements.txt       # Dependencies
```

---

## Camera Intrinsics

For custom datasets, set camera intrinsics in `DatasetCustomTest`:

```python
# core/deepL/datasets/dataset_custom_test.py
def get_camera_parameters(self, idx: int):
    return CameraIntrinsicParameters(
        focal_length_x,      # fx
        focal_length_y,      # fy
        principal_point_x,   # cx
        principal_point_y    # cy
    ), None
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{han2026unicalib,
  title={UniCalib: Targetless LiDAR-camera Calibration via Probabilistic Flow on Unified Depth Representations},
  author={Han, Shu and Zhu, Xubo and Wu, Ji and Cai, Ximeng and Yang, Wen and Yu, Huai and Xia, Gui-Song},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1906--1915},
  year={2026}
}
```

---

## License

MIT License
