# Verification Record

Local checks were performed on 6 September 2026 using a separate Windows
CPU-only test environment. No lab environment or previous experiment was edited.

- CPython 3.12, CPU PyTorch 2.5.1, Lightning 2.4.0, TorchMetrics 1.5.0,
  timm 1.0.11, Hydra 1.3.2 and NumPy 1.26.4.
- Regression tests cover source generation and syntax, pinned-input hashes,
  exact reference YAML composition, global-batch/epoch/warm-up settings,
  all 20 QKNorm sites, combined geometry masks, fully masked rows with finite
  backward gradients, three-window combined forward/backward wiring,
  optimizer parameter coverage, checkpoint fallback without deletion,
  append-preserved/rotated logs, token encodings, credential/size gates,
  bounded full-log snapshot generation, metric summaries and isolated,
  idempotent experiment setup.
- A synthetic-scene CPU Lightning fit/save/resume integration check also runs
  the production callbacks and verifies steps 2 to 4, validation epochs [0, 1]
  without a duplicated partial epoch, and saved optimizer/scheduler state with
  the unchanged 80-epoch schedule. This is a software test, not an experiment result.
- Shell syntax validation covers the launcher and complete user bootstrap.

The CPU combined-forward test substitutes a shape-preserving module for Mamba's
CUDA mixer. It does **not** validate Mamba CUDA execution or reproduce a lab
training result. The actual run refuses to proceed without torch 2.1.1/cu121,
Mamba 1.2.2 and the other pinned previous-run dependencies.

The mandatory on-device preflight performs real AV2 forward/backward,
validation, optimiser updates and full checkpoint restoration on two GPUs.
It is run by the user's launch command, not claimed as already executed from
the laptop. The remaining risks include unusually large scenes, hardware or
filesystem failures, network availability and non-identical interrupted-epoch
replay. Neither uninterrupted execution nor improved final accuracy is guaranteed.
