import sys
import warnings
import loguru
import wandb
from easydict import EasyDict as edict

class Logger:
    """Advanced logger with stderr, log file and Wandb support.
    
    When DistributedDataParallel is enabled, only ERROR logs are activated for slave processes.
    """
    def __init__(self, cfg: edict, log_file=None):
        is_master_node = True
        self._logger = loguru.logger
        self._logger.remove()
        fmt_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level><n>{message}</n></level>"
        log_level = "DEBUG" if is_master_node else "ERROR"
        self._logger.add(sys.stderr, format=fmt_str, colorize=True, level=log_level)
        self._logger.info("Command executed: " + " ".join(sys.argv))
        self._log_file = log_file if is_master_node else None
        if self._log_file is not None:
            fmt_str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            self._logger.add(self._log_file, format=fmt_str, level="INFO")
            self._logger.info(f"Logs are saved to {self._log_file}.")
        wandb.init(
            project = cfg.title,
            dir = cfg.experiment.project_dir,
            name = cfg.experiment.experiment_name,
            config = cfg
        )
        
    @property
    def log_file(self):
        return self._log_file

    def log(self, message, level="INFO"):
        if level not in ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            self._logger.warning(f"Unsupported logging level: {level}. Fallback to INFO.")
            level = "INFO"
        self._logger.log(level, message)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)

    def success(self, message):
        self._logger.success(message)

    def warn(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def critical(self, message):
        self._logger.critical(message)

    def wandb_watch(self, model):
        wandb.watch(model)
    
    def wandb_log(self, data_dict):
        wandb.log(data_dict)

_LOGGER = None

def get_logger(cfg=None, log_file=None):
    """Guarantee only one logger per node is built."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(cfg, log_file=log_file)
    elif log_file is not None:
        log_strings = []
        if log_file is not None:
            log_strings.append(f"log_file={log_file}")
        message = "Logger is already initialized. New parameters (" + ",".join(log_strings) + ") are ignored."
        warnings.warn(message)
    return _LOGGER

def get_print_format(value):
    if isinstance(value, (int, str, tuple)):
        return ""
    if value == 0:
        return ".3f"
    if value < 1e-5:
        return ".3e"
    if value < 1e-2:
        return ".6f"
    return ".3f"

def get_format_strings(result_dict):
    """Get format string for a list of key-value pairs."""
    format_strings = []
    if "metadata" in result_dict:
        # handle special key "metadata"
        format_strings.append(result_dict["metadata"])
    for key, value in result_dict.items():
        if key == "metadata":
            continue
        if isinstance(value, (tuple)):
            format_string = f"{key}: " + "/".join([f"{item:{get_print_format(item)}}" for item in value])
        else:
            format_string = f"{key}: {value:{get_print_format(value)}}"
        format_strings.append(format_string)
    return format_strings

def get_log_string(
    result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, time_dict=None
):
    log_strings = []
    if epoch is not None:
        epoch_string = f"epoch: {epoch}"
        if max_epoch is not None:
            epoch_string += f"/{max_epoch}"
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f"iter: {iteration}"
        if max_iteration is not None:
            iter_string += f"/{max_iteration}"
        log_strings.append(iter_string)
    log_strings += get_format_strings(result_dict)
    if lr is not None:
        log_strings.append("lr: {:.3e}".format(lr))
    if time_dict is not None:
        time_string = "time: " + "/".join([f"{time_dict[key]:.3f}s" for key in time_dict])
        log_strings.append(time_string)
    message = ", ".join(log_strings)
    return message