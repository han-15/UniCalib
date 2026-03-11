from easydict import EasyDict as edict
from pathlib import Path
import datetime
from .io import ensure_dir, read_toml_file
from .parser import parse_args
from .singleton import SingletonType

_CONFIG = None

def get_deafult_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG

def get_config():
    return get_deafult_config().cfg

class Config(metaclass=SingletonType):
    def __init__(self):
        self.cfg = read_toml_file(parse_args().cfg)
        self.add_experiment_cfg()
    
    def add_experiment_cfg(self):
        """
        Adds experiment configuration details to the given configuration dictionary.

        Returns:
            edict: The updated configuration dictionary with added experiment details.

        The function performs the following actions:
            - Sets the experiment name to the title from the configuration.
            - Sets the experiment time to the current datetime in the format YYYYMMDD_HHMMSS.
            - Sets the working directory to the parent directory of the given filename.
            - Sets the project directory to a subdirectory named after the title within the working directory.
            - Sets the output directory to a subdirectory named after the experiment name within the working directory.
            - Sets the checkpoint directory to a "checkpoints" subdirectory within the output directory.
            - Sets the log directory to a "logs" subdirectory within the output directory.
            - Ensures that all directories ending with "_dir" exist by creating them if necessary.
        """
        if "experiment" not in self.cfg:
            self.cfg.experiment = edict()
        self.cfg.experiment.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg.experiment.working_dir = Path(parse_args().cfg).resolve().parent.parent # ./working_dir/cfg/cfg.toml
        self.cfg.experiment.project_dir = self.cfg.experiment.working_dir / self.cfg.title
        self.cfg.experiment.name_dir = self.cfg.experiment.project_dir / ("train" if self.cfg.mode == "train" else "test")
        self.cfg.experiment.experiment_name = self.cfg.mode + "_" + self.cfg.experiment.time
        self.cfg.experiment.output_dir = self.cfg.experiment.name_dir / self.cfg.experiment.experiment_name
        self.cfg.experiment.checkpoint_dir = self.cfg.experiment.output_dir / "checkpoints"
        self.cfg.experiment.log_dir = self.cfg.experiment.output_dir / "logs"
        self.cfg.experiment.result_dir = self.cfg.experiment.output_dir / "result"
        for dir in self.cfg.experiment:
            if dir.endswith("_dir"):
                ensure_dir(self.cfg.experiment[dir])

    def __str__(self) -> str:
        return f"Configuration details:\n{self.cfg}"