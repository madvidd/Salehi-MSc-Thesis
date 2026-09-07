import logging
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    output_dir = cfg.output_dir
    logger.info(f"Experiments are stored in {output_dir}")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    datamodule = instantiate(cfg.datamodule.pl_module, logger=logger)

    model = instantiate(cfg.model.pl_module)
    os.system('cp -a %s %s' % ('src/model', output_dir))
    logger.info(model)

    callbacks = instantiate(cfg.callbacks)
    trainer = pl.Trainer(
        callbacks=callbacks,
        **cfg.trainer
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())

    if trainer.is_global_zero:
        marker = Path(output_dir) / "TRAINING_COMPLETE"
        marker.write_text(
            f"completed_epoch={trainer.current_epoch}\n"
            f"global_step={trainer.global_step}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
