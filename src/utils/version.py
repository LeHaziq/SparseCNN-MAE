import warnings

import torch


def warn_if_not_torch_210() -> None:
    version = torch.__version__.split("+")[0]
    if not version.startswith("2.10"):
        warnings.warn(
            f"Expected PyTorch 2.10.x for reference setup, found {torch.__version__}. "
            "Code supports PyTorch 2.x but behavior/performance may differ.",
            stacklevel=2,
        )
