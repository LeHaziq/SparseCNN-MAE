import logging
import pathlib
import sys
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def get_logger(name: str, log_file: Optional[str] = None, enabled: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    if not enabled:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.disabled = True
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def create_tb_writer(log_dir: Optional[str]) -> SummaryWriter | NullSummaryWriter:
    if not log_dir:
        return NullSummaryWriter()
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
