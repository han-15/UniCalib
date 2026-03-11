import os
from functools import wraps
import torch
import torch.distributed as dist
from torch import Tensor

def setup_distributed(gpus: list):
    """
    Check if the environment variable "LOCAL_RANK" exists. If it exists, use this environment variable 
    to set the GPU device for the current process and initialize the distributed process group. 
    If it does not exist, set the GPU device to cuda:0.
    
    Args:
        gpus: List of GPU indices.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        # single-gpu mode, use cuda:0
        if len(gpus) == 1:
            torch.cuda.set_device(*gpus)

def is_distributed() -> bool:
    """
    Check if the distributed environment is available and initialized.
    
    Returns:
        True if distributed environment is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    """
    Return the size of the distributed environment (i.e., the number of processes participating 
    in distributed computation). If the current environment is not distributed, return 1.
    
    Returns:
        Number of processes in the distributed environment.
    """
    if not is_distributed():
        return 1
    return dist.get_world_size()

def get_local_rank() -> int:
    """
    Return the rank of the current process in the distributed environment. 
    If the current environment is not distributed, return 0.
    
    Returns:
        Rank of the current process.
    """
    if not is_distributed():
        return 0
    return dist.get_rank()

def is_master() -> bool:
    """
    Check if the current process is the master process (i.e., the process with rank 0).
    
    Returns:
        True if the current process is the master process, False otherwise.
    """
    return get_local_rank() == 0

def master_only(func):
    """
    Decorator that ensures the decorated function is only executed in the master process.
    
    Args:
        func: Function to be decorated.
        
    Returns:
        Wrapped function that only executes on the master process.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)

    return wrapper

# reduce tensor
def all_reduce_tensor(tensor, world_size=None):
    """Average reduce a tensor across all workers."""
    if world_size is None:
        world_size = get_world_size()
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor

def all_reduce_tensors(x, world_size=None):
    """Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = tuple([all_reduce_tensors(item, world_size=world_size) for item in x])
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x
