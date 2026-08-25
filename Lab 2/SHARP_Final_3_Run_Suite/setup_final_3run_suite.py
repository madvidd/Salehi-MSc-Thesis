#!/usr/bin/env python3
"""Create an isolated, resumable final three-run SHARP AV2 suite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
OFFICIAL_REPOSITORY = "https://github.com/a-pru/sharp.git"
OFFICIAL_COMMIT = "f6bf2fc0109f9838cdc24bfb763b5c3e6847c2ae"
SUITE_VERSION = "final-sharp-three-run-v1"
VARIANTS = (
    ("01_official_sharp_baseline", "official_sharp_baseline"),
    ("02_qknorm_uncertainty_geometry", "qknorm_uncertainty_geometry"),
    (
        "03_qknorm_uncertainty_geometry_temporal_mamba",
        "qknorm_uncertainty_geometry_temporal_mamba",
    ),
)
PINNED_HASHES = {
    "requirements.txt": "22f04b14376bbb7ab29a6fe7335147b2c560192bd83304f7c7c7698b3313d732",
    "conf/config.yaml": "fee0500585504105a1220ddae3dc19a948c8834dba986da8e6d4f8f0e9592b8d",
    "conf/datamodule/av2_stream.yaml": "b8afe28be0f4bd78bc99b0bdf8b1b1b2e4b76d395436ec2ed7d3b74336a29957",
    "conf/model/Sharp_av2.yaml": "dd11494c10cda12d2d50ee5460aa5e5b3965400aac84b4f282fdb5ef9a4a0a93",
    "src/datamodules/av2_datamodule.py": "5ff29a127795f83742f4b6f907c530f6283d0f0a9aac7421d3bb0f55af5c4148",
    "src/datamodules/av2_dataset.py": "d352ac5da1a996f5747b8b09cf4669dcb8f35adcef3e98cb85c6a7d1594e6f44",
    "src/model/sharp.py": "b048301ea446fd532fe111e84f040ce94187262203ba2b0248eca2894ce5dcf6",
    "src/model/pl_modules.py": "28dd44ef3d21aec97e3c05276565e19506ac78457534e0c3d0315c00c421a7e0",
    "src/model/layers/transformer_blocks.py": "28d7b17242a64249ed466cf317bdf74456c85fcf3f3c3330ef9cd62323c5bbaa",
    "src/model/layers/custom_transformer_blocks.py": "4974ceac16cb087f59f162725753d8f0324a5824e3f3eb254fa47008976ae8cb",
    "src/model/layers/multimodal_decoder_attn.py": "b2800901fe7f32c6aab63fd91ee237329a879d769f0bd1e161a97a0b7ed7543f",
    "train.py": "599637c73b62515da4edead42bdbf0ec942290eff847bfeb4ecfa1e2b0037274",
}


def write_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def clean_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
        "GIT_CEILING_DIRECTORIES",
        "GIT_EXEC_PATH",
        "GIT_TEMPLATE_DIR",
    ):
        environment.pop(name, None)
    return environment


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        env=clean_subprocess_environment(),
        check=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_official_source(base: Path) -> Path:
    cache = base / "Codes" / f"SHARP_OFFICIAL_{OFFICIAL_COMMIT[:12]}"
    git = Path("/usr/bin/git")
    git_command = str(git if git.is_file() else "git")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                git_command,
                "clone",
                "--no-checkout",
                "--filter=blob:none",
                OFFICIAL_REPOSITORY,
                str(cache),
            ]
        )
        run(
            [git_command, "fetch", "--depth", "1", "origin", OFFICIAL_COMMIT],
            cwd=cache,
        )
        run([git_command, "checkout", "--detach", "FETCH_HEAD"], cwd=cache)
    if not (cache / ".git").is_dir():
        raise RuntimeError(f"Official-source cache is not a Git clone: {cache}")
    head = subprocess.check_output(
        [git_command, "rev-parse", "HEAD"],
        cwd=cache,
        env=clean_subprocess_environment(),
        text=True,
    ).strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"Official-source cache is at {head}, expected {OFFICIAL_COMMIT}"
        )
    dirty = subprocess.check_output(
        [git_command, "status", "--porcelain"],
        cwd=cache,
        env=clean_subprocess_environment(),
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            "Pinned official-source cache has local modifications; refusing to "
            f"build an inexact baseline:\n{dirty}"
        )
    mismatches = []
    for relative, expected in PINNED_HASHES.items():
        actual = sha256(cache / relative)
        if actual != expected:
            mismatches.append(f"{relative}: {actual} != {expected}")
    if mismatches:
        raise RuntimeError("Pinned official source failed hash audit:\n" + "\n".join(mismatches))
    return cache


def find_mamba_packages(base: Path, python_bin: Path) -> Path:
    candidates: list[Path] = []
    for pointer_name in (
        "LATEST_SHARP_AV2_MAMBA_FUSED80.txt",
        "LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80.txt",
    ):
        pointer = base / "Codes" / pointer_name
        if pointer.is_file():
            root = Path(pointer.read_text(encoding="utf-8").strip())
            candidates.extend((root / "fused_packages", root / "mamba_packages"))
    candidates.extend(
        item / "fused_packages"
        for item in sorted(
            (base / "Codes").glob("SHARP_AV2_MAMBA_FUSED80_*"), reverse=True
        )
    )
    try:
        completed = subprocess.run(
            [
                str(python_bin),
                "-c",
                (
                    "import pathlib,mamba_ssm; "
                    "print(pathlib.Path(mamba_ssm.__file__).resolve().parent.parent)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates.insert(0, Path(completed.stdout.strip()))
    except (subprocess.CalledProcessError, OSError):
        pass

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        package = candidate / "mamba_ssm"
        scans = list(candidate.glob("selective_scan_cuda*.so"))
        if package.is_dir() and scans:
            return candidate.resolve()
    raise RuntimeError(
        "No verified fused mamba_ssm/selective_scan_cuda package was found. "
        "The completed Lab 2 fused-Mamba experiment must remain available."
    )


def patch_mask_compatibility(code_dir: Path) -> None:
    helper = '''

def _match_attention_mask_dtypes(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
):
    """Return equivalent masks with the dtype contract required by PyTorch."""
    if (
        attn_mask is None
        or key_padding_mask is None
        or attn_mask.dtype == key_padding_mask.dtype
    ):
        return attn_mask, key_padding_mask
    if attn_mask.is_floating_point() and key_padding_mask.dtype == torch.bool:
        converted = torch.zeros_like(key_padding_mask, dtype=attn_mask.dtype)
        return attn_mask, converted.masked_fill(key_padding_mask, float("-inf"))
    if key_padding_mask.is_floating_point() and attn_mask.dtype == torch.bool:
        converted = torch.zeros_like(attn_mask, dtype=key_padding_mask.dtype)
        return converted.masked_fill(attn_mask, float("-inf")), key_padding_mask
    if attn_mask.is_floating_point() and key_padding_mask.is_floating_point():
        return attn_mask, key_padding_mask.to(dtype=attn_mask.dtype)
    raise TypeError(
        "Unsupported attention-mask dtype pair: "
        f"{attn_mask.dtype}, {key_padding_mask.dtype}"
    )
'''
    for filename in (
        "custom_transformer_blocks.py",
        "transformer_blocks.py",
    ):
        path = code_dir / "src/model/layers" / filename
        source = path.read_text(encoding="utf-8")
        source = replace_once(
            source,
            "from timm.models.layers import DropPath",
            "from timm.layers import DropPath",
            f"timm import in {filename}",
        )
        source = replace_once(
            source,
            "from timm.layers import DropPath\n",
            "from timm.layers import DropPath\n" + helper,
            f"mask helper in {filename}",
        )
        calls = 0
        for call in (
            "        src2 = self.attn(\n",
            "        attn_output = self.attn(\n",
        ):
            count = source.count(call)
            calls += count
            source = source.replace(
                call,
                (
                    "        mask, key_padding_mask = "
                    "_match_attention_mask_dtypes(mask, key_padding_mask)\n"
                    + call
                ),
            )
        if calls == 0:
            raise RuntimeError(f"No attention calls found in {path}")
        write_lf(path, source)


def patch_common(code_dir: Path) -> None:
    model_cfg_path = code_dir / "conf/model/Sharp_av2.yaml"
    model_cfg = model_cfg_path.read_text(encoding="utf-8")
    model_cfg = replace_once(model_cfg, "    lr: 1e-3\n", "    lr: 1e-4\n", "paper LR")
    model_cfg = replace_once(
        model_cfg,
        "    warmup_ratio: 0.167\n",
        "    warmup_ratio: 0.1625\n",
        "13-of-80 warmup ratio",
    )
    write_lf(model_cfg_path, model_cfg)

    config_path = code_dir / "conf/config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = replace_once(config, "gpus: 1\n", "gpus: 4\n", "GPU count")
    config = replace_once(config, "batch_size: 32\n", "batch_size: 8\n", "per-rank batch")
    config = replace_once(config, "epochs: 60\n", "epochs: 80\n", "paper epochs")
    config = replace_once(
        config,
        "    save_top_k: 10\n",
        "    save_top_k: 10\n    save_last: True\n    every_n_epochs: 1\n",
        "resumable checkpoint settings",
    )
    callback_anchor = (
        "  - _target_: pytorch_lightning.callbacks.LearningRateMonitor\n"
        "    logging_interval: epoch\n"
    )
    config = replace_once(
        config,
        callback_anchor,
        callback_anchor
        + "  - _target_: src.utils.metrics_history_callback.MetricsHistoryCallback\n"
        + "    output_path: ${output_dir}/metrics_history.jsonl\n",
        "metrics-history callback",
    )
    config = replace_once(
        config,
        "  accelerator: gpu\n",
        "  accelerator: gpu\n  precision: 32-true\n",
        "explicit FP32 precision",
    )
    write_lf(config_path, config)

    shutil.copy2(
        PACKAGE_ROOT / "runtime/metrics_history_callback.py",
        code_dir / "src/utils/metrics_history_callback.py",
    )
    patch_mask_compatibility(code_dir)

    sharp_path = code_dir / "src/model/sharp.py"
    sharp = sharp_path.read_text(encoding="utf-8")
    sharp = replace_once(
        sharp,
        "torch.inverse(rot_mat)",
        "rot_mat.transpose(1, 2)",
        "orthogonal rotation inverse",
    )
    write_lf(sharp_path, sharp)

    pl_path = code_dir / "src/model/pl_modules.py"
    pl_source = pl_path.read_text(encoding="utf-8")
    for tensor_name in ("y_hat", "new_y_hat"):
        old = f"torch.arange({tensor_name}.shape[0])"
        pl_source = pl_source.replace(
            old,
            f"torch.arange({tensor_name}.shape[0], device={tensor_name}.device)",
        )
    if pl_source.count("device=y_hat.device") != 2 or pl_source.count(
        "device=new_y_hat.device"
    ) != 1:
        raise RuntimeError("CUDA-safe loss-index patch count is incorrect")
    pl_source = replace_once(
        pl_source,
        "            batch_size=1,\n",
        "            batch_size=len(data[-1][\"scenario_id\"]),\n",
        "stream validation metric batch size",
    )
    pl_source = replace_once(
        pl_source,
        "        #assert len(param_dict.keys() - union_params) == 0\n",
        "        missing_parameters = param_dict.keys() - union_params\n"
        "        assert not missing_parameters, (\n"
        "            f'Trainable parameters missing from AdamW groups: {sorted(missing_parameters)}'\n"
        "        )\n",
        "optimizer coverage audit",
    )
    write_lf(pl_path, pl_source)

    datamodule_path = code_dir / "src/datamodules/av2_datamodule.py"
    datamodule = datamodule_path.read_text(encoding="utf-8")
    anchor = "            pin_memory=self.pin_memory,\n            collate_fn=collate_fn,\n"
    replacement = (
        "            pin_memory=self.pin_memory,\n"
        "            persistent_workers=self.num_workers > 0,\n"
        "            prefetch_factor=4 if self.num_workers > 0 else None,\n"
        "            collate_fn=collate_fn,\n"
    )
    count = datamodule.count(anchor)
    if count != 3:
        raise RuntimeError(f"Expected three DataLoader anchors, found {count}")
    write_lf(datamodule_path, datamodule.replace(anchor, replacement))


def patch_qknorm(code_dir: Path) -> None:
    shutil.copy2(
        PACKAGE_ROOT / "runtime/qknorm_attention.py",
        code_dir / "src/model/layers/qknorm_attention.py",
    )
    total = 0
    for filename in (
        "custom_transformer_blocks.py",
        "transformer_blocks.py",
    ):
        path = code_dir / "src/model/layers" / filename
        source = path.read_text(encoding="utf-8")
        source = replace_once(
            source,
            "from torch import Tensor\n",
            "from torch import Tensor\nfrom .qknorm_attention import QKNormMultiheadAttention\n",
            f"QKNorm import in {filename}",
        )
        count = source.count("torch.nn.MultiheadAttention(")
        total += count
        source = source.replace(
            "torch.nn.MultiheadAttention(", "QKNormMultiheadAttention("
        )
        write_lf(path, source)
    if total != 3:
        raise RuntimeError(f"Expected three SHARP attention constructors, found {total}")


def patch_enhancements(code_dir: Path, with_mamba: bool) -> None:
    shutil.copy2(
        PACKAGE_ROOT / "runtime/final_enhancements.py",
        code_dir / "src/model/layers/final_enhancements.py",
    )
    if with_mamba:
        shutil.copy2(
            PACKAGE_ROOT / "runtime/temporal_agent_mamba.py",
            code_dir / "src/model/layers/temporal_agent_mamba.py",
        )

    path = code_dir / "src/model/sharp.py"
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "from typing import List\n\n",
        "from typing import List\n\nimport os\n\n",
        "variant environment import",
    )
    source = replace_once(
        source,
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n",
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n"
        "from .layers.final_enhancements import (\n"
        "    RelativeGeometryBias,\n"
        "    UncertaintyTargetContext,\n"
        ")\n",
        "enhancement imports",
    )
    source = replace_once(
        source,
        "        self.future_steps = future_steps\n        self.dm = dm\n",
        "        self.future_steps = future_steps\n"
        "        self.dm = dm\n"
        "        self.final_suite_variant = os.environ.get(\n"
        "            'SHARP_FINAL_VARIANT', 'official_sharp_baseline'\n"
        "        )\n",
        "final-suite variant selection",
    )
    module_block = '''        self.relative_geometry_bias = RelativeGeometryBias(num_heads)
        self.uncertainty_target_context = UncertaintyTargetContext()

        self.initialize_weights()
        self.temporal_agent_mamba = None
'''
    if with_mamba:
        module_block += '''        from .layers.temporal_agent_mamba import TemporalAgentMamba
        self.temporal_agent_mamba = TemporalAgentMamba(
            dim=embed_dim,
            d_state=8,
            d_conv=3,
            expand=1,
            dropout=0.1,
            layer_scale_init=0.01,
            agent_chunk_size=128,
        )
'''
    module_block += "        return\n"
    source = replace_once(
        source,
        "        self.initialize_weights()\n        return\n",
        module_block,
        "composed enhancement modules",
    )
    source = replace_once(
        source,
        "        for blk in self.h_embed:\n            actor_feat = blk(actor_feat, key_padding_mask=kpm)\n",
        "        for block_index, blk in enumerate(self.h_embed):\n"
        "            actor_feat = blk(actor_feat, key_padding_mask=kpm)\n"
        "            if self.temporal_agent_mamba is not None and block_index == 1:\n"
        "                actor_feat = self.temporal_agent_mamba(actor_feat, ~kpm)\n",
        "small residual temporal Mamba insertion",
    )
    source = replace_once(
        source,
        "        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),\n                                lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()\n",
        "        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),\n"
        "                                lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()\n"
        "        scene_attention_bias = self.relative_geometry_bias(\n"
        "            torch.cat([data['x_centers'], data['lane_centers']], dim=1),\n"
        "            torch.cat([data['x_angles'][:, :, -1], data['lane_angles']], dim=1),\n"
        "        )\n",
        "relative geometry attention bias",
    )
    source = replace_once(
        source,
        "            max_distance = 30\n            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed\n            target_mask = (~key_valid_mask_all.unsqueeze(1).expand(B, self.k, key_valid_mask_all.shape[1])) | (torch.norm(x_centers, dim=-1) > max_distance)\n",
        "            max_distance = target_pos.new_full((B, self.k), 30.0)\n"
        "            target_feature_gate = target_pos.new_ones((B, self.k))\n"
        "            if 'pi' in data['memory_dict']:\n"
        "                max_distance, target_feature_gate = self.uncertainty_target_context(\n"
        "                    data['memory_dict']['pi'].float()\n"
        "                )\n"
        "            else:\n"
        "                uncertainty_ddp_zero = target_pos.new_zeros(())\n"
        "                for parameter in self.uncertainty_target_context.parameters():\n"
        "                    uncertainty_ddp_zero = uncertainty_ddp_zero + parameter.sum() * 0.0\n"
        "                target_feature_gate = target_feature_gate + uncertainty_ddp_zero\n"
        "            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed\n"
        "            target_mask = (~key_valid_mask_all.unsqueeze(1).expand(B, self.k, key_valid_mask_all.shape[1])) | (torch.norm(x_centers, dim=-1) > max_distance.unsqueeze(-1))\n",
        "uncertainty-aware target radius",
    )
    source = replace_once(
        source,
        "            compressed_target_encoder = compressed_target_encoder.view(B, self.k, -1, self.embed_dim) + target_center_embed.unsqueeze(2)\n",
        "            compressed_target_encoder = compressed_target_encoder.view(B, self.k, -1, self.embed_dim) + target_center_embed.unsqueeze(2)\n"
        "            compressed_target_encoder = compressed_target_encoder * target_feature_gate.unsqueeze(-1).unsqueeze(-1)\n",
        "uncertainty-aware feature gate",
    )
    source = replace_once(
        source,
        "            container[target_valid.view(-1)] = compressed_target_encoder.view(-1, self.embed_dim)[~compressed_target_mask.view(-1)]\n",
        "            target_destination = torch.nonzero(\n"
        "                target_valid.reshape(-1), as_tuple=False\n"
        "            ).squeeze(-1)\n"
        "            compressed_source = torch.nonzero(\n"
        "                ~compressed_target_mask.reshape(-1), as_tuple=False\n"
        "            ).squeeze(-1)\n"
        "            if target_destination.numel() != compressed_source.numel():\n"
        "                raise RuntimeError(\n"
        "                    'Target-context remap count mismatch: '\n"
        "                    f'destination={target_destination.numel()} '\n"
        "                    f'source={compressed_source.numel()}'\n"
        "                )\n"
        "            compressed_flat = compressed_target_encoder.reshape(\n"
        "                -1, self.embed_dim\n"
        "            ).index_select(0, compressed_source)\n"
        "            container = torch.index_copy(\n"
        "                container, 0, target_destination, compressed_flat\n"
        "            )\n",
        "CUDA-safe target-context remap",
    )
    source = source.replace(
        "                x_curr = blk(x_curr, key_padding_mask=~key_valid_mask)\n",
        "                x_curr = blk(x_curr, mask=scene_attention_bias, key_padding_mask=~key_valid_mask)\n",
        1,
    )
    source = replace_once(
        source,
        "            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)\n",
        "            x_encoder = blk(x_encoder, mask=scene_attention_bias, key_padding_mask=~key_valid_mask)\n",
        "geometry-biased scene attention",
    )
    source = replace_once(
        source,
        "                'cache_ids': torch.cat([data['agent_ids'], data['lane_ids']], dim=1)\n",
        "                'cache_ids': torch.cat([data['agent_ids'], data['lane_ids']], dim=1),\n"
        "                'pi': pi.detach()\n",
        "streamed mode probabilities",
    )
    write_lf(path, source)

    pl_path = code_dir / "src/model/pl_modules.py"
    pl_source = pl_path.read_text(encoding="utf-8")
    pl_source = replace_once(
        pl_source,
        "        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + new_agent_reg_loss + new_agent_cls_loss\n",
        "        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + new_agent_reg_loss + new_agent_cls_loss\n"
        "        for parameter in self.model.uncertainty_target_context.parameters():\n"
        "            loss = loss + parameter.sum() * 0.0\n",
        "DDP-safe uncertainty parameter bridge",
    )
    write_lf(pl_path, pl_source)


def patch_train(code_dir: Path, variant: str) -> None:
    path = code_dir / "train.py"
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "import os\n",
        "import os\nfrom pathlib import Path\n",
        "Path import",
    )
    audit = f'''    variant = os.environ.get("SHARP_FINAL_VARIANT", "")
    if variant != "{variant}":
        raise RuntimeError(f"Expected SHARP_FINAL_VARIANT={variant}, got {{variant}}")
    names = [name for name, _ in model.named_parameters()]
    has_qknorm = any("logit_scale" in name for name in names)
    has_geometry = any("relative_geometry_bias" in name for name in names)
    has_uncertainty = any("uncertainty_target_context" in name for name in names)
    has_mamba = any("temporal_agent_mamba" in name for name in names)
    expected = {{
        "official_sharp_baseline": (False, False, False, False),
        "qknorm_uncertainty_geometry": (True, True, True, False),
        "qknorm_uncertainty_geometry_temporal_mamba": (True, True, True, True),
    }}[variant]
    actual = (has_qknorm, has_geometry, has_uncertainty, has_mamba)
    if actual != expected:
        raise RuntimeError(f"Architecture audit failed: expected={{expected}}, actual={{actual}}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FINAL_SUITE_ARCHITECTURE_OK={{variant}} parameters={{trainable}}")
    Path(output_dir, "ARCHITECTURE_AUDIT.txt").write_text(
        f"variant={{variant}}\\ntrainable_parameters={{trainable}}\\n"
        f"qknorm={{has_qknorm}}\\ngeometry={{has_geometry}}\\n"
        f"uncertainty={{has_uncertainty}}\\nmamba={{has_mamba}}\\n"
    )
'''
    source = replace_once(
        source,
        "    model = instantiate(cfg.model.pl_module)\n",
        "    model = instantiate(cfg.model.pl_module)\n" + audit,
        "runtime architecture audit",
    )
    source = replace_once(
        source,
        "    trainer.validate(model, datamodule.val_dataloader())\n",
        '''    if not bool(cfg.trainer.get("fast_dev_run", False)):
        best_path = trainer.checkpoint_callback.best_model_path
        if not best_path:
            raise RuntimeError("No best checkpoint was recorded")
        Path(output_dir, "BEST_CHECKPOINT.txt").write_text(best_path + "\\n")
        print(f"FINAL_BEST_CHECKPOINT={best_path}")
''',
        "single-GPU post-training evaluation handoff",
    )
    write_lf(path, source)


def create_variant(source: Path, destination: Path, variant: str) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".hydra", "outputs"
        ),
    )
    patch_common(destination)
    if variant != "official_sharp_baseline":
        patch_qknorm(destination)
        patch_enhancements(
            destination,
            with_mamba=variant.endswith("temporal_mamba"),
        )
    patch_train(destination, variant)
    write_lf(
        destination / "FINAL_SUITE_VARIANT.txt",
        f"suite_version={SUITE_VERSION}\nvariant={variant}\n",
    )


def reusable_experiment(base: Path) -> Path | None:
    pointer = base / "Codes/LATEST_SHARP_FINAL_3RUN_CODE.txt"
    if not pointer.is_file():
        return None
    candidate = Path(pointer.read_text(encoding="utf-8").strip())
    marker = candidate / "SUITE_MANIFEST.json"
    runner = candidate / "run_final_3run_suite.sh"
    if not marker.is_file() or not runner.is_file():
        return None
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("suite_version") != SUITE_VERSION:
        return None
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/server00/M")
    parser.add_argument("--force-new", action="store_true")
    args = parser.parse_args()
    base = Path(args.base).resolve()
    python_bin = base / "Codes/envs/sharp/bin/python"
    dataset = base / "Datasets/AV2/sharp_processed"
    token_file = base / "Token/Token.txt"
    for required in (python_bin, dataset / "train", dataset / "val", token_file):
        if not required.exists():
            raise SystemExit(f"Missing required Lab 2 path: {required}")

    existing = None if args.force_new else reusable_experiment(base)
    if existing is not None:
        manifest = json.loads((existing / "SUITE_MANIFEST.json").read_text())
        results = Path(manifest["results_root"])
        write_lf(base / "Codes/LATEST_SHARP_FINAL_3RUN_CODE.txt", str(existing) + "\n")
        write_lf(base / "Results/LATEST_SHARP_FINAL_3RUN_RESULTS.txt", str(results) + "\n")
        print("REUSING_RESUMABLE_FINAL_SUITE")
        print(f"EXPERIMENT_ROOT={existing}")
        print(f"RESULTS_ROOT={results}")
        print(f"RUNNER={existing / 'run_final_3run_suite.sh'}")
        return

    source = ensure_official_source(base)
    mamba_packages = find_mamba_packages(base, python_bin)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_root = base / "Codes" / f"SHARP_FINAL_3RUN_{stamp}"
    results_root = base / "Results" / f"SHARP_FINAL_3RUN_{stamp}"
    experiment_root.mkdir(parents=True, exist_ok=False)
    results_root.mkdir(parents=True, exist_ok=False)

    for slug, variant in VARIANTS:
        create_variant(source, experiment_root / "variants" / slug / "Code", variant)
        (results_root / slug).mkdir(parents=True)

    mamba_link = experiment_root / "mamba_packages"
    os.symlink(mamba_packages, mamba_link, target_is_directory=True)
    runtime_patch = experiment_root / "runtime_patch"
    runtime_patch.mkdir()
    shutil.copy2(PACKAGE_ROOT / "runtime/sitecustomize.py", runtime_patch / "sitecustomize.py")

    copied_scripts = (
        "run_final_3run_suite.sh",
        "preflight_final_suite.py",
        "eval_checkpoint.py",
        "generate_final_artifacts.py",
        "publish_final_artifacts.sh",
    )
    for filename in copied_scripts:
        target = experiment_root / filename
        shutil.copy2(PACKAGE_ROOT / filename, target)
        if target.suffix in (".sh", ".py"):
            make_executable(target)

    environment = "\n".join(
        (
            f"BASE={shlex.quote(str(base))}",
            f"EXPERIMENT_ROOT={shlex.quote(str(experiment_root))}",
            f"RESULTS_ROOT={shlex.quote(str(results_root))}",
            f"OFFICIAL_SOURCE={shlex.quote(str(source))}",
            f"OFFICIAL_COMMIT={shlex.quote(OFFICIAL_COMMIT)}",
            f"MAMBA_PACKAGES={shlex.quote(str(mamba_link))}",
            f"RUNTIME_PATCH={shlex.quote(str(runtime_patch))}",
            f"SUITE_VERSION={shlex.quote(SUITE_VERSION)}",
        )
    )
    write_lf(experiment_root / "suite.env", environment + "\n")
    manifest = {
        "suite_version": SUITE_VERSION,
        "created": dt.datetime.now().astimezone().isoformat(),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_source": str(source),
        "experiment_root": str(experiment_root),
        "results_root": str(results_root),
        "dataset": str(dataset),
        "python": str(python_bin),
        "paper_schedule": {
            "epochs": 80,
            "warmup_epochs": 13,
            "global_batch": 32,
            "per_gpu_batch": 8,
            "gpus": 4,
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.00001,
            "weight_decay": 0.01,
            "gradient_clip_norm": 5,
            "seed": 2333,
            "precision": "32-true",
        },
        "variants": [
            {"slug": slug, "variant": variant} for slug, variant in VARIANTS
        ],
    }
    write_lf(
        experiment_root / "SUITE_MANIFEST.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    write_lf(
        experiment_root / "PINNED_SOURCE_SHA256.txt",
        "\n".join(f"{value}  {key}" for key, value in PINNED_HASHES.items())
        + "\n",
    )
    write_lf(base / "Codes/LATEST_SHARP_FINAL_3RUN_CODE.txt", str(experiment_root) + "\n")
    write_lf(base / "Results/LATEST_SHARP_FINAL_3RUN_RESULTS.txt", str(results_root) + "\n")

    print("FINAL_3RUN_SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"RUNNER={experiment_root / 'run_final_3run_suite.sh'}")
    print("Previous code and results were not changed or deleted.")


if __name__ == "__main__":
    main()
