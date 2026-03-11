"""
DatasetCustomTest Usage Example

This file demonstrates how to use the DatasetCustomTest class to load custom test data.
The dataset automatically scans the images folder and matches corresponding depth images and point cloud files.
"""

from easydict import EasyDict as edict
from core.constant import CameraIntrinsicParameters, EngineMode
from core.deepL.datasets import DatasetCustomTest

# ============================================================================
# Example 1: Basic Usage
# ============================================================================

def example_basic_usage():
    """Basic usage example"""
    
    # 1. Prepare configuration parameters (passed via cfg)
    cfg = edict({
        'root_folder': '/path/to/your/data',           # Data root directory
        'pcl_folder': 'pointclouds',                  # Point cloud folder (relative path)
        'imgs_folder': 'images',                       # RGB image folder (relative path)
        'depth_imgs_folder': 'depth_images',           # Depth image folder (relative path)
        'w_scale': 1.0,                                # Width scale ratio
        'h_scale': 1.0,                                # Height scale ratio
        'adjust_coordinate_order': 'xyz',              # Coordinate order adjustment (optional)
        'max_r': 10.0,                                 # Maximum rotation angle (degrees)
        'max_t': 2.0,                                  # Maximum translation distance (meters)
    })
    
    # 2. Prepare camera intrinsics
    camera_intrinsic = CameraIntrinsicParameters(
        focal_length_x=718.856,      # fx
        focal_length_y=718.856,      # fy
        principal_point_x=607.1928,  # cx
        principal_point_y=185.2157   # cy
    )
    
    # 3. Create dataset instance (no longer requires a JSON file)
    dataset = DatasetCustomTest(
        cfg=cfg,
        camera_intrinsic=camera_intrinsic,
        engine_mode=EngineMode.TEST
    )
    
    # 4. Use the dataset
    print(f"Dataset size: {len(dataset)}")
    
    # Get the first sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Depth image shape: {sample['vision_image'].shape}")
    print(f"Original image shape: {sample['original_image'].shape}")
    print(f"Point cloud shape: {sample['point_cloud'].shape}")
    
    return dataset


# ============================================================================
# Example 2: Usage with Extra Parameters
# ============================================================================

def example_with_extra_params():
    """Usage example with extra custom parameters"""
    
    cfg = edict({
        'root_folder': '/path/to/your/data',
        'pcl_folder': 'pointclouds',
        'imgs_folder': 'images',
        'depth_imgs_folder': 'depth_images',
        'w_scale': 0.5,  # Scale to 50% of original size
        'h_scale': 0.5,
        'adjust_coordinate_order': 'xyz',
        'max_r': 10.0,
        'max_t': 2.0,
    })
    
    camera_intrinsic = CameraIntrinsicParameters(
        focal_length_x=718.856,
        focal_length_y=718.856,
        principal_point_x=607.1928,
        principal_point_y=185.2157
    )
    
    # Can pass in extra custom parameters
    dataset = DatasetCustomTest(
        cfg=cfg,
        camera_intrinsic=camera_intrinsic,
        engine_mode=EngineMode.TEST,
        # The following are extra custom parameters
        custom_param1="value1",
        custom_param2=42,
        custom_param3=True
    )
    
    # Access custom parameters
    print(f"Custom parameters: {dataset.user_params}")
    
    return dataset


# ============================================================================
# Example 3: Batch Loading with DataLoader
# ============================================================================

def example_with_dataloader():
    """Example of batch loading data with DataLoader"""
    
    from torch.utils.data import DataLoader
    from core.deepL.datasets.dataset import merge_inputs_with_original
    
    cfg = edict({
        'root_folder': '/path/to/your/data',
        'pcl_folder': 'pointclouds',
        'imgs_folder': 'images',
        'depth_imgs_folder': 'depth_images',
        'w_scale': 1.0,
        'h_scale': 1.0,
        'adjust_coordinate_order': 'xyz',
        'max_r': 10.0,
        'max_t': 2.0,
    })
    
    camera_intrinsic = CameraIntrinsicParameters(
        focal_length_x=718.856,
        focal_length_y=718.856,
        principal_point_x=607.1928,
        principal_point_y=185.2157
    )
    
    dataset = DatasetCustomTest(
        cfg=cfg,
        camera_intrinsic=camera_intrinsic,
        engine_mode=EngineMode.TEST
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=merge_inputs_with_original
    )
    
    # Iterate over data
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Depth image batch shape: {batch_data['vision_image'][0].shape}")
        print(f"  Point cloud count: {len(batch_data['point_cloud'])}")
        break  # Only print the first batch
    
    return dataloader


# ============================================================================
# File Naming Conventions
# ============================================================================

"""
File naming conventions:

The dataset automatically scans all image files in the images folder and
matches corresponding depth images and point cloud files by filename (without extension).

Examples:
- images/000000.jpg      -> depth_images/000000.npy + pointclouds/000000.bin
- images/000001.png      -> depth_images/000001.npy + pointclouds/000001.bin
- images/frame_001.jpg   -> depth_images/frame_001.npy + pointclouds/frame_001.bin

Notes:
1. Image filenames (without extension) must match depth image and point cloud filenames
2. Supported RGB image formats: .jpg, .png, .jpeg
3. Depth images must be in .npy format
4. Point cloud files must be in .bin format
5. If an image file cannot find its corresponding depth image or point cloud file, the sample will be skipped
"""


# ============================================================================
# Directory Structure Example
# ============================================================================

"""
Recommended directory structure:

/path/to/your/data/
├── images/                  # RGB image folder (the dataset will scan this folder)
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── 000002.jpg
├── depth_images/            # Depth image folder
│   ├── 000000.npy
│   ├── 000001.npy
│   └── 000002.npy
└── pointclouds/             # Point cloud folder
    ├── 000000.bin
    ├── 000001.bin
    └── 000002.bin

Note: A data_list.json file is no longer needed!
The dataset automatically scans the images folder and matches corresponding depth images and point clouds by filename.
"""


if __name__ == '__main__':
    print("=" * 80)
    print("DatasetCustomTest Usage Example")
    print("=" * 80)
    
    # Note: Before running these examples, modify the paths to your actual data paths
    
    # example_basic_usage()
    # example_with_extra_params()
    # example_with_dataloader()
    
    print("\nPlease modify the example code with your actual data paths before running.")
