import abc
import torch

from pathlib import Path

from ..checkpoint import load_state_dict
from ..setup_engine import setup_engine
from ..metrics_manager import MetricsManager
from ....utils import (
    get_default_parser,
    get_logger,
    get_log_string,
    get_config
)

class BaseTester(abc.ABC):
    def __init__(self):
        # parser
        parser = get_default_parser()
        self._args = parser.parse_args()
        self._cudnn_deterministic = self._args.cudnn_deterministic

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."
        
        cfg = get_config()
        self._cfg = cfg
        # logger
        self._log_file = cfg.experiment.log_dir / "test.log"
        self._logger = get_logger(cfg, self._log_file)

        # find checkpoint
        self._checkpoint = self._args.checkpoint
        assert Path(self._checkpoint).exists(), f"Checkpoint not found: {self._checkpoint}"
        
        # metrics manager
        self._metrics_manager = MetricsManager()
        
        # initialize
        torch.cuda.set_device(*cfg.gpus)
        setup_engine(seed=cfg.experiment.seed, cudnn_deterministic=self._cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        # data loader
        self.test_loader = None

    @property
    def args(self):
        return self._args

    @property
    def log_file(self):
        return self._log_file

    def load(self, filename, strict=True):
        self.log('Loading from "{}".'.format(filename))
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        assert "model" in state_dict, "No model can be loaded."
        load_state_dict(self.model, state_dict["model"], strict=strict)
        self.log("Model has been loaded.")
        if "metadata" in state_dict:
            epoch = state_dict["metadata"]["epoch"]
            total_steps = state_dict["metadata"]["total_steps"]
            self.log(f"Checkpoint metadata: epoch: {epoch}, total_steps: {total_steps}.")

    def register_model(self, model):
        """Register model."""
        model = model.cuda()
        self.model = model
        message = "Model description:\n" + str(model)
        self.log(message)
        return model

    def register_loader(self, test_loader):
        """Register data loader."""
        self.test_loader = test_loader

    def log(self, message, level="INFO"):
        self._logger.log(message, level=level)

    def write_dict(self, data_dict):
        """Write Wandb event."""
        self._logger.wandb_log(data_dict)
        
    def metrics_clear(self):
        self._metrics_manager.clear()
        
    def metrics_update(self, data_dict):
        self._metrics_manager.update(data_dict)

    def metrics_summary_mean(self):
        return self._metrics_manager.get_metrics_mean()
    
    def metrics_summary_mean_std(self):
        return self._metrics_manager.get_metrics_mean_std()

    def before_test_epoch(self):
        self.metrics_clear()

    def before_test_step(self, iteration, data_dict):
        return data_dict

    @abc.abstractmethod
    def test_step(self, iteration, data_dict) -> dict:
        pass

    @abc.abstractmethod
    def eval_step(self, iteration, data_dict, output_dict) -> dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self, summary_dict):
        pass

    def get_log_string(self, iteration, data_dict, output_dict, result_dict) -> str:
        return get_log_string(result_dict)

    @abc.abstractmethod
    def test_epoch(self):
        pass

    def run(self, strict_loading=True):
        assert self.test_loader is not None
        if self._checkpoint is not None:
            self.load(self._checkpoint, strict=strict_loading)
        else:
            self.log("Checkpoint is not specified.", level="WARNING")
        total_parameters = sum(p.numel() for p in self.model.parameters())
        self.log(f"Total parameters: {total_parameters}.")
        self.model.eval()
        torch.set_grad_enabled(False)
        self.test_epoch()


