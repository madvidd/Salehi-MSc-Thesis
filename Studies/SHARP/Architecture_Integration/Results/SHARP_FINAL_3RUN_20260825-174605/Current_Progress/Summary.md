# Lab 2 Final SHARP Three-Run Progress

- Captured: 2026-08-31T21:03:49+01:00
- Results root: `/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-174605`
- Completed variants: 1/3
- Active variant: `02_qknorm_uncertainty_geometry`
- Suite complete: False
- Complete local transcript: `/home/server00/M/Terminal/SHARP_Final_3_Run_Suite/SHARP_FINAL_3RUN_20260825-174605/Snapshots/Progress_20260831-210327/Terminal.txt` (174634770 bytes)
- GitHub Terminal.txt mode: compact transcript; complete transcript retained locally
- Append-only local Terminal.txt: `/home/server00/M/Terminal/SHARP_Final_3_Run_Suite/SHARP_FINAL_3RUN_20260825-174605/Terminal.txt` (174635208 bytes); previously saved lines were retained.

## Progress And Best Validation Metrics

All metrics are lower-is-better. Best values are selected by the lowest recorded minADE6 epoch.

| Run | Status | Current progress | Latest validated epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 | Checkpoints |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official SHARP baseline | completed | epoch 79, 100% | 79 | 0.1560 | 1.9246 | 1.6769 | 0.6793 | 4.1386 | 1.2877 | 11 |
| SHARP + QKNorm + uncertainty + geometry | running | epoch 0, 7% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Run 2 + residual temporal-agent Mamba | pending | not started | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |

## Runtime Diagnostics

- Matched warning/error lines in the complete transcript: 18 retained (latest 200 maximum).
- Snapshot generation only read logs and `/proc`; it did not send signals to training.

```text
Epoch 0:   0%|          | 11/6248 [00:09<1:27:42,  1.19it/s, v_num=0]Error executing job with overrides: ['seed=2333', 'gpus=4', 'batch_size=8', 'epochs=80', 'output_dir=/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-174605/02_qknorm_uncertainty_geometry/run', 'datamodule.pl_module.data_root=/home/server00/M/Datasets/AV2/sharp_processed', 'datamodule.pl_module.num_workers=4', 'model.pl_module.optim.lr=0.0001', 'model.pl_module.optim.min_lr=0.00001', 'model.pl_module.optim.warmup_ratio=0.1625', 'model.pl_module.optim.weight_decay=0.01', 'trainer.devices=4', 'trainer.strategy=ddp_find_unused_parameters_false', 'trainer.sync_batchnorm=true', 'trainer.precision=32-true']
Traceback (most recent call last):
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Traceback (most recent call last):
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
[2026-08-31 20:31:24] server00:1055210:1055845 [2] misc/strongstream.cc:399 NCCL WARN Cuda failure 'an illegal memory access was encountered'
[2026-08-31 20:31:24] server00:1055210:1055845 [2] init.cc:2018 NCCL WARN commDestroySync: comm 0x5e6cadeaeda0 rank 2 sync hostStream error 1
[2026-08-31 20:31:24] server00:1055210:1055845 [2] misc/strongstream.cc:399 NCCL WARN Cuda failure 'an illegal memory access was encountered'
[2026-08-31 20:31:24] server00:1055210:1055845 [2] init.cc:2021 NCCL WARN commDestroySync: comm 0x5e6cadeaeda0 rank 2 sync deviceStream error 1
Epoch 0:   1%|          | 56/6248 [00:42<1:17:54,  1.32it/s, v_num=1]Error executing job with overrides: ['seed=2333', 'gpus=4', 'batch_size=8', 'epochs=80', 'output_dir=/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-174605/02_qknorm_uncertainty_geometry/run', 'datamodule.pl_module.data_root=/home/server00/M/Datasets/AV2/sharp_processed', 'datamodule.pl_module.num_workers=4', 'model.pl_module.optim.lr=0.0001', 'model.pl_module.optim.min_lr=0.00001', 'model.pl_module.optim.warmup_ratio=0.1625', 'model.pl_module.optim.weight_decay=0.01', 'trainer.devices=4', 'trainer.strategy=ddp_find_unused_parameters_false', 'trainer.sync_batchnorm=true', 'trainer.precision=32-true']
Traceback (most recent call last):
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Traceback (most recent call last):
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
[2026-08-31 20:48:48] server00:1062821:1063846 [2] misc/strongstream.cc:399 NCCL WARN Cuda failure 'an illegal memory access was encountered'
[2026-08-31 20:48:48] server00:1062821:1063846 [2] init.cc:2018 NCCL WARN commDestroySync: comm 0x5997fb663110 rank 2 sync hostStream error 1
[2026-08-31 20:48:48] server00:1062821:1063846 [2] misc/strongstream.cc:399 NCCL WARN Cuda failure 'an illegal memory access was encountered'
[2026-08-31 20:48:48] server00:1062821:1063846 [2] init.cc:2021 NCCL WARN commDestroySync: comm 0x5997fb663110 rank 2 sync deviceStream error 1
```
