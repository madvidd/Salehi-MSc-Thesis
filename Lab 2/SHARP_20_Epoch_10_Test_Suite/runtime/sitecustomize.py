"""Compatibility fixes for the pinned Lab 2 SHARP environment."""

import warnings

import numpy as np


for _name, _value in (("bool", bool), ("int", int), ("float", float)):
    if _name not in np.__dict__:
        setattr(np, _name, _value)

# These notices are emitted by dependencies and do not indicate changed behavior.
warnings.filterwarnings(
    "ignore", message=r"The 'repr' attribute with value False was provided.*"
)
warnings.filterwarnings(
    "ignore", message=r"The 'frozen' attribute with value True was provided.*"
)
warnings.filterwarnings(
    "ignore", message=r"No device id is provided via .*init_process_group.*"
)
