#!/usr/bin/env python3
"""Finalize Mamba initialization and make each training attempt isolated."""

from pathlib import Path


base = Path("/home/server00/M")
experiment_root = Path(
    (base / "Codes/LATEST_SHARP_AV2_MAMBA_ENCODER_V2.txt")
    .read_text(encoding="utf-8")
    .strip()
)
layer_path = experiment_root / "Code/src/model/layers/mamba_encoder.py"
sharp_path = experiment_root / "Code/src/model/sharp.py"
run_path = experiment_root / "run_av2_mamba_4gpu.sh"

layer = layer_path.read_text(encoding="utf-8")
old_init = '''        # Stable Mamba-style delta initialization in [0.001, 0.1].
        dt = torch.exp(
'''
new_init = '''        self.reset_dt_bias()

    def reset_dt_bias(self) -> None:
        """Restore the stable Mamba delta initialization after SHARP init."""
        dt = torch.exp(
'''
if old_init not in layer:
    raise SystemExit("Mamba delta initialization anchor not found")
layer = layer.replace(old_init, new_init, 1)

old_scene = '''        self.residual_gates = nn.ParameterList(
            nn.Parameter(torch.tensor(-2.1972246)) for _ in range(depth)
        )

    def forward(
'''
new_scene = '''        self.residual_gates = nn.ParameterList(
            nn.Parameter(torch.tensor(-2.1972246)) for _ in range(depth)
        )

    def reset_mamba_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_dt_bias()

    def forward(
'''
if old_scene not in layer:
    raise SystemExit("Scene Mamba initialization anchor not found")
layer_path.write_text(layer.replace(old_scene, new_scene, 1), encoding="utf-8")

sharp = sharp_path.read_text(encoding="utf-8")
old_sharp = '''        self.initialize_weights()
        return
'''
new_sharp = '''        self.initialize_weights()
        # SHARP reinitializes Linear layers, so restore Mamba's delta bias.
        self.mamba_encoder.reset_mamba_parameters()
        return
'''
if old_sharp not in sharp:
    raise SystemExit("SHARP initialization anchor not found")
sharp_path.write_text(sharp.replace(old_sharp, new_sharp, 1), encoding="utf-8")

run_script = run_path.read_text(encoding="utf-8")
old_results = next(
    line for line in run_script.splitlines() if line.startswith('RESULTS_DIR="')
)
new_results = (
    old_results.replace("RESULTS_DIR=", "RESULTS_ROOT=")
    + '\nATTEMPT_ID=$(date +%Y%m%d-%H%M%S)'
    + '\nRESULTS_DIR="$RESULTS_ROOT/$ATTEMPT_ID"'
)
run_path.write_text(
    run_script.replace(old_results, new_results, 1),
    encoding="utf-8",
)

print(f"MAMBA_EXPERIMENT_FINALIZED={experiment_root}")
