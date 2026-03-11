from easydict import EasyDict as edict
from torch.utils.data.dataloader import default_collate
from .dataloader import build_dataloader

from ...utils import get_logger
from ...constant import EngineMode

datasets = {}


def register_dataset(cls):
    """
    Register a dataset class.
    
    Args:
        cls: Dataset class to register.
        
    Returns:
        The registered dataset class.
    """
    datasets[cls.__name__] = cls
    return cls


def merge_inputs(queries):
    """
    Merge input data for DataLoader's collate_fn.
    
    Args:
        queries: List of data dictionaries from dataset.
        
    Returns:
        Merged dictionary with batched data.
    """
    point_clouds = []
    imgs = []
    calibs = []
    orders = []
    returns = {
        key: default_collate([d[key] for d in queries]) 
        for key in queries[0] 
        if key not in ['point_cloud', 'vision_image', 'camera_intrinsic_parameters', 'order']
    }
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['vision_image'])
        calibs.append(input['camera_intrinsic_parameters'])
        orders.append(input['order'])
    
    returns['point_cloud'] = point_clouds
    returns['vision_image'] = imgs
    returns['camera_intrinsic_parameters'] = calibs
    returns['order'] = orders
    return returns


def merge_inputs_with_original(queries):
    """
    Merge input data including original images for testing collate_fn.
    
    Args:
        queries: List of data dictionaries from dataset.
        
    Returns:
        Merged dictionary with batched data including original images.
    """
    point_clouds = []
    imgs = []
    calibs = []
    orders = []
    original_imgs = []
    returns = {
        key: default_collate([d[key] for d in queries]) 
        for key in queries[0]
        if key not in ['point_cloud', 'original_image', 'vision_image', 'camera_intrinsic_parameters', 'order']
    }
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['vision_image'])
        calibs.append(input['camera_intrinsic_parameters'])
        orders.append(input['order'])
        original_imgs.append(input['original_image'])
    
    returns['point_cloud'] = point_clouds
    returns['vision_image'] = imgs
    returns['camera_intrinsic_parameters'] = calibs
    returns['order'] = orders
    returns['original_image'] = original_imgs
    return returns

def create_dataset(cfg: edict = None, engine_mode: EngineMode = None):
    """
    Create a dataset instance.
    
    Args:
        cfg: Configuration dictionary containing dataset parameters.
        engine_mode: Engine mode (TRAIN, VALID, or TEST).
        
    Returns:
        Dataset instance.
    """
    assert cfg is not None, 'cfg must be provided to create dataset'
    assert 'dataset' in cfg or 'name' in cfg, 'dataset must be provided to create dataset'
    assert engine_mode is not None, 'engine_mode must be provided to create dataset'
    
    if 'dataset' in cfg:
        assert 'name' in cfg['dataset'], 'dataset name must be provided to create dataset'
        assert cfg['dataset']['name'] in datasets, f"dataset {cfg['dataset']['name']} is not registered"
        dataset = datasets[cfg['dataset']['name']](cfg, engine_mode)
        return dataset
    else:
        dataset = datasets[cfg['name']](cfg, engine_mode)
        return dataset

def get_train_valid_data_loader(cfg: edict) -> tuple:
    """
    Get training and validation data loaders.
    
    Args:
        cfg: Configuration dictionary.
        
    Returns:
        Tuple of (train_data_loader, valid_data_loader).
    """
    get_logger().info('Loading training and validation data loaders...')
    train_dataset = create_dataset(cfg.dataset, engine_mode=EngineMode.TRAIN)
    valid_dataset = create_dataset(cfg.dataset, engine_mode=EngineMode.VALID)
    
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty.")
    if len(valid_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    train_data_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.dataset['batch_size'],
        num_workers=cfg.dataset['num_workers'],
        shuffle=True,
        collate_fn=merge_inputs,
    )
    valid_data_loader = build_dataloader(
        valid_dataset,
        num_workers=cfg.dataset['num_workers'],
        shuffle=False,
        collate_fn=merge_inputs,
    )
    return train_data_loader, valid_data_loader

def get_test_data_loader(cfg: edict):
    """
    Get test data loader.
    
    Args:
        cfg: Configuration dictionary.
        
    Returns:
        Test data loader.
    """
    get_logger().info('Loading test data loader...')
    test_dataset = create_dataset(cfg.dataset, engine_mode=EngineMode.TEST)
    test_data_loader = build_dataloader(
        test_dataset,
        num_workers=cfg.dataset['num_workers'],
        batch_size=1,
        shuffle=False,
        collate_fn=merge_inputs_with_original
    )
    return test_data_loader