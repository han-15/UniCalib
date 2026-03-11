import abc
import torch.nn as nn
from easydict import EasyDict as edict

class Evaluation(nn.Module, abc.ABC):
    def __init__(self, cfg: edict):
        self._cfg = cfg
        super(Evaluation, self).__init__()
        
    @abc.abstractmethod
    def evaluation_fn(self, data_dict: dict, output_dict: dict) -> dict:
        raise NotImplementedError
    
    def forward(self, data_dict: dict, output_dict: dict):
        result_dict = self.evaluation_fn(data_dict, output_dict)
        return result_dict