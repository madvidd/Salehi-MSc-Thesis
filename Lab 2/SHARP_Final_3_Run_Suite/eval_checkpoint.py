#!/usr/bin/env python3
"""Evaluate one final-suite checkpoint on one GPU without DDP padding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def serializable(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value) if isinstance(value, (int, float)) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    code = args.code.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint is missing: {checkpoint}")

    pl.seed_everything(2333, workers=True)
    with initialize_config_dir(version_base=None, config_dir=str(code / "conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "gpus=1",
                "batch_size=32",
                f"datamodule.pl_module.data_root={args.dataset.resolve()}",
                "datamodule.pl_module.num_workers=4",
            ],
        )
    datamodule = instantiate(cfg.datamodule.pl_module)
    model = instantiate(cfg.model.pl_module)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        default_root_dir=str(output.parent),
    )
    result = trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint),
        verbose=True,
    )
    if len(result) != 1:
        raise RuntimeError(f"Expected one validation result, got {len(result)}")
    payload = {key: serializable(value) for key, value in result[0].items()}
    payload["checkpoint"] = str(checkpoint)
    payload["evaluation_devices"] = 1
    payload["evaluation_batch_size"] = 32
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"FINAL_EVALUATION_JSON={output}")


if __name__ == "__main__":
    main()
