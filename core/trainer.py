import time
from .model import *
from .evaluation import *
from core.deepL.engine import EpochBasedTrainer, build_optimizer, build_scheduler
from core.deepL.datasets.dataset import (
    get_train_valid_data_loader,
)
from core.deepL.model import create_model
from core.deepL.evaluation import get_evaluation
from core.deepL.datasets.data_preprocess import DepthFlowGenerator
from core.constant.deepL import EngineMode


class Trainer(EpochBasedTrainer):
    def __init__(self):
        super().__init__()
        # Enable CUDA auto-tuning
        torch.backends.cudnn.benchmark = True
        # dataloader
        start_time = time.time()
        train_loader, val_loader = get_train_valid_data_loader(
            self._cfg
        )

        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.register_loader(train_loader, val_loader)
        self._cfg.nums_train_dataset = len(train_loader)
        # model
        model = create_model(self._cfg)
        model = self.register_model(model)

        # optimizer, scheduler
        optimizer = build_optimizer(model, self._cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, self._cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = get_evaluation('SequenceLossFunction', self._cfg)
        self.eval_func = get_evaluation('SequenceEvalFunction', self._cfg)
        self.save_best_model_on_metric(self.eval_func.val_metric, largest=False)
        
        # preparation
        self.depth_flow_generator = DepthFlowGenerator(self._cfg)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict, engine_mode=EngineMode.VALID)
        result_dict = self.eval_func(data_dict, output_dict)
        return output_dict, result_dict

    def before_train_step(self, epoch, iteration, data_dict):
        data_dict = self.depth_flow_generator.push(data_dict)
        return data_dict

    def before_val_step(self, epoch, iteration, data_dict):
        data_dict = self.depth_flow_generator.push(data_dict, EngineMode.VALID)
        return data_dict
    