import json
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    if not cfg.checkpoint or not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {cfg.checkpoint}")

    pl.seed_everything(cfg.seed, workers=True)
    datamodule = instantiate(cfg.datamodule.pl_module)
    model = instantiate(cfg.model.pl_module)
    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=cfg.gpus,
        strategy="ddp_find_unused_parameters_false" if cfg.gpus > 1 else "auto",
        max_epochs=1,
        enable_checkpointing=False,
    )
    result = trainer.validate(model, datamodule, ckpt_path=cfg.checkpoint)

    if trainer.is_global_zero:
        output = Path(os.environ["SHARP_METRICS_JSON"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result[0], indent=2, sort_keys=True) + "\n")
        print(f"METRICS_JSON={output}")


if __name__ == "__main__":
    main()
