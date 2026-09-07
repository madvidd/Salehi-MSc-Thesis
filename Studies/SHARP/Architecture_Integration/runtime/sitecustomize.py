"""Narrow compatibility fixes for the pinned SHARP runtime."""

import warnings

import numpy as np


for _name, _value in (("bool", bool), ("int", int), ("float", float)):
    if _name not in np.__dict__:
        setattr(np, _name, _value)

# PyTorch emits this despite each Lightning rank setting its CUDA device first.
warnings.filterwarnings(
    "ignore",
    message=r"No device id is provided via .*",
)

# Verified metadata-only notices from the pinned Hydra/Lightning dependency set.
warnings.filterwarnings(
    "ignore",
    message=r"The 'repr' attribute with value False was provided.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The 'frozen' attribute with value True was provided.*",
)

# Four workers per DDP rank provide 16 workers in total. Lightning's warning
# compares each rank with all host CPUs and therefore overstates this setting.
warnings.filterwarnings(
    "ignore",
    message=r"The '.*_dataloader' does not have many workers.*",
)
