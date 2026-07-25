# Lab 2: SHARP with Mamba Encoder Addition

Dataset: Argoverse 2 motion forecasting
Hardware: 4 x RTX 2080 Ti
Epochs: 80
Batch size: 8 per GPU; global batch size 32
Synchronized BatchNorm: enabled
Modification: fused bidirectional Mamba encoder

Best checkpoint: epoch 78
MR: 0.1573532075
b-minFDE6: 1.9264185429
minADE1: 1.6803642511
minADE6: 0.6803619266
minFDE1: 4.1300029755
minFDE6: 1.2904220819

Large checkpoints, archives, Terminal.txt and full logs remain local.
Local archive: /home/server00/M/Results/Archives/Lab2_SHARP_Mamba_final_20260724-180557
