"""Rank-zero validation-history callback for reproducible result reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from pytorch_lightning import Callback


METRICS = (
    "MR",
    "b-minFDE6",
    "minADE1",
    "minADE6",
    "minFDE1",
    "minFDE6",
)


class MetricsHistoryCallback(Callback):
    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        record: dict[str, int | float] = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }
        for name in METRICS:
            value = trainer.callback_metrics.get(name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            record[name] = float(value)
        if len(record) == 2:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
