# Source Provenance and Isolation

The bundled `upstream/seam-main` tree is an isolated copy of the SEAM AV2 source that produced the previous verified baseline run. The baseline model instantiated from this package contains `4,604,769` trainable parameters, exactly matching that earlier baseline. The preflight enforces this count before training.

The controlled changes are confined to these files:

- `src/model/layers/controlled_ablation.py`: QKNorm, uncertainty control, and relative-geometry bias modules.
- `src/model/layers/custom_transformer_blocks.py`: optional QKNorm for temporal and decoder attention.
- `src/model/layers/transformer_blocks.py`: optional QKNorm for scene and streaming attention.
- `src/model/layers/multimodal_decoder_attn.py`: forwards the QKNorm selection through decoder attention.
- `src/model/seam.py`: activates exactly one named intervention and places it in the audited SEAM location.
- `src/model/pl_modules.py`: supplies explicit validation batch sizes to Lightning logging.
- `conf/config.yaml`: applies the common 20-epoch Lab 3 execution budget and per-epoch checkpoint retention.

The previous SEAM package, its 80-epoch results, and every earlier experiment pointer remain unchanged. Setup copies this package to a new timestamped code directory and writes results to a new timestamped result directory.
