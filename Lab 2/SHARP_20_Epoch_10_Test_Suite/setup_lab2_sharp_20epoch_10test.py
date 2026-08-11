#!/usr/bin/env python3
"""Prepare an isolated, resumable ten-variant SHARP AV2 experiment on Lab 2."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import stat
import subprocess


PACKAGE_ROOT = Path(__file__).resolve().parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("\r\n", "\n"))


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def find_mamba_packages(base: Path, python_bin: Path) -> Path:
    candidates: list[Path] = []
    pointer = base / "Codes/LATEST_SHARP_AV2_MAMBA_FUSED80.txt"
    if pointer.is_file():
        candidates.append(Path(pointer.read_text(encoding="utf-8").strip()) / "fused_packages")
    candidates.extend(
        item / "fused_packages"
        for item in sorted(base.joinpath("Codes").glob("SHARP_AV2_MAMBA_FUSED80_*"), reverse=True)
    )
    try:
        completed = subprocess.run(
            [
                str(python_bin),
                "-c",
                "import pathlib,mamba_ssm; print(pathlib.Path(mamba_ssm.__file__).resolve().parent.parent)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        candidates.append(Path(completed.stdout.strip()))
    except (OSError, subprocess.CalledProcessError):
        pass

    for candidate in candidates:
        if candidate.is_dir() and candidate.joinpath("mamba_ssm").is_dir():
            if any(candidate.glob("selective_scan_cuda*.so")):
                return candidate.resolve()
    raise SystemExit(
        "A verified mamba_ssm package with selective_scan_cuda was not found. "
        "Keep the completed SHARP_AV2_MAMBA_FUSED80 experiment available."
    )


def patch_attention_masks(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source = source.replace(
        "from timm.models.layers import DropPath", "from timm.layers import DropPath"
    )
    helper = '''\n\ndef _match_attention_mask_dtypes(\n    attn_mask: Optional[torch.Tensor],\n    key_padding_mask: Optional[torch.Tensor],\n):\n    """Match equivalent attention-mask dtypes required by current PyTorch."""\n    if (\n        attn_mask is None\n        or key_padding_mask is None\n        or attn_mask.dtype == key_padding_mask.dtype\n    ):\n        return attn_mask, key_padding_mask\n    if attn_mask.is_floating_point() and key_padding_mask.dtype == torch.bool:\n        converted = torch.zeros_like(key_padding_mask, dtype=attn_mask.dtype)\n        return attn_mask, converted.masked_fill(key_padding_mask, float("-inf"))\n    if key_padding_mask.is_floating_point() and attn_mask.dtype == torch.bool:\n        converted = torch.zeros_like(attn_mask, dtype=key_padding_mask.dtype)\n        return converted.masked_fill(attn_mask, float("-inf")), key_padding_mask\n    if attn_mask.is_floating_point() and key_padding_mask.is_floating_point():\n        return attn_mask, key_padding_mask.to(dtype=attn_mask.dtype)\n    raise TypeError(\n        f"Unsupported attention-mask dtypes: {attn_mask.dtype}, {key_padding_mask.dtype}"\n    )\n'''
    source = replace_once(
        source,
        "from timm.layers import DropPath\n",
        "from timm.layers import DropPath\n" + helper,
        f"attention helper import in {path.name}",
    )
    call_count = 0
    for call in ("        src2 = self.attn(\n", "        attn_output = self.attn(\n"):
        count = source.count(call)
        call_count += count
        source = source.replace(
            call,
            "        mask, key_padding_mask = _match_attention_mask_dtypes(\n"
            "            mask, key_padding_mask\n"
            "        )\n"
            + call,
        )
    if call_count == 0:
        raise RuntimeError(f"No attention calls were patched in {path}")
    write_lf(path, source)


def patch_common(code_dir: Path) -> None:
    model_cfg_path = code_dir / "conf/model/Sharp_av2.yaml"
    model_cfg = model_cfg_path.read_text(encoding="utf-8")
    model_cfg = replace_once(model_cfg, "    lr: 1e-3\n", "    lr: 1e-4\n", "paper LR")
    model_cfg = replace_once(
        model_cfg,
        "    warmup_ratio: 0.167\n",
        "    warmup_ratio: 0.65\n",
        "13-of-20 warm-up ratio",
    )
    write_lf(model_cfg_path, model_cfg)

    config_path = code_dir / "conf/config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = replace_once(config, "gpus: 1\n", "gpus: 4\n", "GPU count")
    config = replace_once(config, "batch_size: 32\n", "batch_size: 8\n", "per-GPU batch")
    config = replace_once(config, "epochs: 60\n", "epochs: 20\n", "epoch count")
    config = replace_once(config, "    save_top_k: 10\n", "    save_top_k: -1\n    save_last: True\n    every_n_epochs: 1\n", "epoch checkpointing")
    config = replace_once(
        config,
        "trainer:\n",
        "trainer:\n  num_sanity_val_steps: 0\n",
        "trainer preflight setting",
    )
    write_lf(config_path, config)

    for layer_name in ("custom_transformer_blocks.py", "transformer_blocks.py"):
        patch_attention_masks(code_dir / "src/model/layers" / layer_name)

    sharp_path = code_dir / "src/model/sharp.py"
    sharp = sharp_path.read_text(encoding="utf-8")
    sharp = sharp.replace("torch.inverse(rot_mat)", "rot_mat.transpose(1, 2)")
    write_lf(sharp_path, sharp)

    pl_path = code_dir / "src/model/pl_modules.py"
    pl = pl_path.read_text(encoding="utf-8")
    pl = pl.replace("torch.inverse(memory_dict[\"rot_mat\"])", "memory_dict[\"rot_mat\"].transpose(1, 2)")
    pl = replace_once(
        pl,
        '''        self.log_dict(\n            reg_loss_dict,\n            on_step=False,\n            on_epoch=True,\n            prog_bar=False,\n            sync_dist=True,\n        )\n''',
        '''        self.log_dict(\n            reg_loss_dict,\n            on_step=False,\n            on_epoch=True,\n            prog_bar=False,\n            sync_dist=True,\n            batch_size=len(data[-1]["scenario_id"]),\n        )\n''',
        "stream validation loss batch size",
    )
    pl = replace_once(pl, "            batch_size=1,\n", "            batch_size=len(data[-1][\"scenario_id\"]),\n", "stream validation metric batch size")
    write_lf(pl_path, pl)

    datamodule_path = code_dir / "src/datamodules/av2_datamodule.py"
    datamodule = datamodule_path.read_text(encoding="utf-8")
    needle = "            pin_memory=self.pin_memory,\n            collate_fn=collate_fn,\n"
    replacement = (
        "            pin_memory=self.pin_memory,\n"
        "            persistent_workers=self.num_workers > 0,\n"
        "            prefetch_factor=4 if self.num_workers > 0 else None,\n"
        "            collate_fn=collate_fn,\n"
    )
    count = datamodule.count(needle)
    if count < 2:
        raise RuntimeError(f"Expected train/val DataLoader anchors, found {count}")
    datamodule = datamodule.replace(needle, replacement)
    write_lf(datamodule_path, datamodule)


def patch_sharp(code_dir: Path) -> None:
    path = code_dir / "src/model/sharp.py"
    source = path.read_text(encoding="utf-8")
    if "mamba" in source.lower():
        raise RuntimeError("Original source already contains Mamba; refusing to stack changes")

    source = replace_once(
        source,
        "from typing import List\n\n",
        "from typing import List\n\nimport os\n\n",
        "os import",
    )
    source = replace_once(
        source,
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n",
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n"
        "from .layers.ablation_modules import (\n"
        "    EndpointRefinementHead,\n"
        "    GeometricLaneGraphRefiner,\n"
        "    KinematicMotionStem,\n"
        "    LearnedTemporalPooling,\n"
        "    MemoryUpdateGate,\n"
        "    RelativeGeometryBias,\n"
        "    UncertaintyTargetContext,\n"
        "    VARIANTS,\n"
        ")\n",
        "ablation imports",
    )
    source = replace_once(
        source,
        "        self.future_steps = future_steps\n        self.dm = dm\n",
        "        self.future_steps = future_steps\n"
        "        self.dm = dm\n"
        "        self.experiment_variant = os.environ.get(\n"
        "            'SHARP_EXPERIMENT_VARIANT', 'baseline'\n"
        "        )\n"
        "        if self.experiment_variant not in VARIANTS:\n"
        "            raise ValueError(f'Unknown experiment variant: {self.experiment_variant}')\n",
        "variant selection",
    )
    original_temporal = '''        self.h_embed = nn.ModuleList(\n            CustomBlock(\n                dim=embed_dim,\n                num_heads=num_heads,\n                mlp_ratio=mlp_ratio,\n                qkv_bias=qkv_bias,\n                drop_path=dpr[i],\n                cross_attn=False\n            )\n            for i in range(encoder_depth)\n        )\n'''
    replacement_temporal = '''        if self.experiment_variant == "agent_temporal_mamba":\n            self.h_embed = nn.ModuleList()\n        else:\n            self.h_embed = nn.ModuleList(\n                CustomBlock(\n                    dim=embed_dim,\n                    num_heads=num_heads,\n                    mlp_ratio=mlp_ratio,\n                    qkv_bias=qkv_bias,\n                    drop_path=dpr[i],\n                    cross_attn=False\n                )\n                for i in range(encoder_depth)\n            )\n'''
    source = replace_once(source, original_temporal, replacement_temporal, "temporal encoder selection")

    source = replace_once(
        source,
        "        self.initialize_weights()\n        return\n",
        '''        self.temporal_pool = (\n            LearnedTemporalPooling(embed_dim)\n            if self.experiment_variant == "learned_temporal_pool" else None\n        )\n        self.kinematic_stem = (\n            KinematicMotionStem(embed_dim)\n            if self.experiment_variant == "kinematic_motion_stem" else None\n        )\n        self.memory_update_gate = (\n            MemoryUpdateGate(embed_dim)\n            if self.experiment_variant == "confidence_gated_memory" else None\n        )\n        self.relative_geometry_bias = (\n            RelativeGeometryBias(num_heads)\n            if self.experiment_variant == "relative_geometry_bias" else None\n        )\n        self.uncertainty_target_context = (\n            UncertaintyTargetContext()\n            if self.experiment_variant == "uncertainty_target_context" else None\n        )\n        self.endpoint_refinement = (\n            EndpointRefinementHead(embed_dim, future_steps)\n            if self.experiment_variant == "endpoint_refinement_decoder" else None\n        )\n        self.lane_graph_refiner = (\n            GeometricLaneGraphRefiner(embed_dim)\n            if self.experiment_variant == "lane_topology_graph" else None\n        )\n\n        self.initialize_weights()\n        self.temporal_mamba = None\n        if self.experiment_variant == "agent_temporal_mamba":\n            from .layers.temporal_mamba_replacement import TemporalMambaStack\n            self.temporal_mamba = TemporalMambaStack(\n                dim=embed_dim, depth=4, d_state=16, d_conv=4, expand=2,\n                drop_path=drop_path, actor_chunk_size=256,\n            )\n        return\n''',
        "variant modules",
    )

    source = replace_once(
        source,
        '''        actor_feat = torch.cat([actor_feat, ts], dim=-1)\n\n        actor_feat = self.h_proj( actor_feat )\n        kpm = (~hist_valid_mask).view(B*N, -1)[hist_feat_key_valid_mask]\n        for blk in self.h_embed:\n            actor_feat = blk(actor_feat, key_padding_mask=kpm)\n        actor_feat = torch.max(actor_feat, axis=1).values\n''',
        '''        actor_feat = torch.cat([actor_feat, ts], dim=-1)\n        actor_raw_features = actor_feat\n\n        actor_feat = self.h_proj(actor_feat)\n        kpm = (~hist_valid_mask).view(B*N, -1)[hist_feat_key_valid_mask]\n        temporal_valid_mask = ~kpm\n        if self.kinematic_stem is not None:\n            actor_feat = self.kinematic_stem(\n                actor_raw_features, actor_feat, temporal_valid_mask\n            )\n        if self.temporal_mamba is not None:\n            actor_feat = self.temporal_mamba(actor_feat, temporal_valid_mask)\n        else:\n            for blk in self.h_embed:\n                actor_feat = blk(actor_feat, key_padding_mask=kpm)\n        if self.temporal_pool is not None:\n            actor_feat = self.temporal_pool(actor_feat, temporal_valid_mask)\n        else:\n            actor_feat = actor_feat.masked_fill(kpm.unsqueeze(-1), float("-inf"))\n            actor_feat = torch.max(actor_feat, axis=1).values\n''',
        "agent encoder forward",
    )
    source = replace_once(
        source,
        "        lane_feat = lane_feat.view(B, M, -1)\n",
        "        lane_feat = lane_feat.view(B, M, -1)\n"
        "        if self.lane_graph_refiner is not None:\n"
        "            lane_feat = self.lane_graph_refiner(\n"
        "                lane_feat, data['lane_centers'], data['lane_angles'],\n"
        "                data['lane_key_valid_mask'],\n"
        "            )\n",
        "lane graph refinement",
    )
    source = replace_once(
        source,
        "        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),\n                                lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()\n",
        "        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),\n"
        "                                lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()\n"
        "        scene_attention_bias = None\n"
        "        if self.relative_geometry_bias is not None:\n"
        "            scene_attention_bias = self.relative_geometry_bias(\n"
        "                torch.cat([data['x_centers'], data['lane_centers']], dim=1),\n"
        "                torch.cat([data['x_angles'][:, :, -1], data['lane_angles']], dim=1),\n"
        "            )\n",
        "relative geometry bias",
    )
    source = replace_once(
        source,
        '''            max_distance = 30\n            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed\n            target_mask = (~key_valid_mask_all.unsqueeze(1).expand(B, self.k, key_valid_mask_all.shape[1])) | (torch.norm(x_centers, dim=-1) > max_distance)\n''',
        '''            max_distance = target_pos.new_full((B, self.k), 30.0)\n            target_feature_gate = target_pos.new_ones((B, self.k))\n            if self.uncertainty_target_context is not None and "pi" in data["memory_dict"]:\n                max_distance, target_feature_gate = self.uncertainty_target_context(\n                    data["memory_dict"]["pi"].float()\n                )\n            elif self.uncertainty_target_context is not None:\n                # Keep conditionally inactive parameters in the DDP graph without\n                # changing forward values before streamed probabilities exist.\n                uncertainty_ddp_zero = target_pos.new_zeros(())\n                for parameter in self.uncertainty_target_context.parameters():\n                    uncertainty_ddp_zero = uncertainty_ddp_zero + parameter.sum() * 0.0\n                target_feature_gate = target_feature_gate + uncertainty_ddp_zero\n            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed\n            target_mask = (~key_valid_mask_all.unsqueeze(1).expand(B, self.k, key_valid_mask_all.shape[1])) | (torch.norm(x_centers, dim=-1) > max_distance.unsqueeze(-1))\n''',
        "uncertainty target radius",
    )
    source = replace_once(
        source,
        "            compressed_target_encoder = compressed_target_encoder.view(B, self.k, -1, self.embed_dim) + target_center_embed.unsqueeze(2)\n",
        "            compressed_target_encoder = compressed_target_encoder.view(B, self.k, -1, self.embed_dim) + target_center_embed.unsqueeze(2)\n"
        "            compressed_target_encoder = compressed_target_encoder * target_feature_gate.unsqueeze(-1).unsqueeze(-1)\n",
        "uncertainty target gate",
    )
    source = source.replace(
        "                x_curr = blk(x_curr, key_padding_mask=~key_valid_mask)\n",
        "                x_curr = blk(x_curr, mask=scene_attention_bias, key_padding_mask=~key_valid_mask)\n",
        1,
    )
    source = replace_once(
        source,
        '''                if self.biased_interaction:\n                    cache_ids = torch.cat([data['agent_ids'], data['lane_ids']], dim=1)\n                    mask = (cache_ids.unsqueeze(2) == memory_cache_ids.unsqueeze(1))\n                    mask = mask[x_type_mask].reshape(B, -1, memory_ids.shape[-1])\n                    # agent to agent+lanes with mask\n                    new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask, mask=mask)           \n                else:\n                    # agent to agent+lanes\n                    new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)\n                # lane to lane\n                new_lane_feat = self.scene_interact(new_x_encoder[~x_type_mask].reshape(B, -1, C), memory_x_encoder[~memory_type_mask].reshape(B, -1, C), cur_pose, memory_pose, key_padding_mask=~memory_valid_mask[~memory_type_mask].reshape(B, -1))\n                new_x_encoder = torch.cat([new_actor_feat, new_lane_feat], dim=1)\n''',
        '''                current_actor_feat = new_x_encoder[x_type_mask].reshape(B, -1, C)\n                if self.biased_interaction:\n                    cache_ids = torch.cat([data['agent_ids'], data['lane_ids']], dim=1)\n                    mask = (cache_ids.unsqueeze(2) == memory_cache_ids.unsqueeze(1))\n                    mask = mask[x_type_mask].reshape(B, -1, memory_ids.shape[-1])\n                    new_actor_feat = self.scene_interact(current_actor_feat, memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask, mask=mask)\n                else:\n                    new_actor_feat = self.scene_interact(current_actor_feat, memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)\n                current_lane_feat = new_x_encoder[~x_type_mask].reshape(B, -1, C)\n                new_lane_feat = self.scene_interact(current_lane_feat, memory_x_encoder[~memory_type_mask].reshape(B, -1, C), cur_pose, memory_pose, key_padding_mask=~memory_valid_mask[~memory_type_mask].reshape(B, -1))\n                if self.memory_update_gate is not None:\n                    new_actor_feat = self.memory_update_gate(current_actor_feat, new_actor_feat)\n                    new_lane_feat = self.memory_update_gate(current_lane_feat, new_lane_feat)\n                new_x_encoder = torch.cat([new_actor_feat, new_lane_feat], dim=1)\n''',
        "confidence-gated streaming",
    )
    source = replace_once(
        source,
        "            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)\n",
        "            x_encoder = blk(x_encoder, mask=scene_attention_bias, key_padding_mask=~key_valid_mask)\n",
        "scene attention bias call",
    )
    source = replace_once(
        source,
        "        y_hat, pi, aux_dec_ret = self.decoder(x_agent, x_encoder, (~key_valid_mask), N, aux=aux)\n        x_mode = aux_dec_ret[0]\n",
        "        y_hat, pi, aux_dec_ret = self.decoder(x_agent, x_encoder, (~key_valid_mask), N, aux=aux)\n"
        "        x_mode = aux_dec_ret[0]\n"
        "        coarse_y_hat = None\n"
        "        coarse_pi = None\n"
        "        if self.endpoint_refinement is not None:\n"
        "            coarse_y_hat, coarse_pi = y_hat, pi\n"
        "            y_hat, pi = self.endpoint_refinement(x_mode, y_hat, pi)\n",
        "endpoint refinement",
    )
    source = replace_once(
        source,
        "            'pi_single': pi_single\n        }\n",
        "            'pi_single': pi_single,\n"
        "            'coarse_y_hat': coarse_y_hat,\n"
        "            'coarse_pi': coarse_pi,\n"
        "        }\n",
        "coarse decoder outputs",
    )
    source = replace_once(
        source,
        "                'cache_ids': torch.cat([data['agent_ids'], data['lane_ids']], dim=1)\n",
        "                'cache_ids': torch.cat([data['agent_ids'], data['lane_ids']], dim=1),\n"
        "                'pi': pi.detach()\n",
        "streamed probability memory",
    )
    write_lf(path, source)


def patch_lightning(code_dir: Path) -> None:
    path = code_dir / "src/model/pl_modules.py"
    source = path.read_text(encoding="utf-8")
    source = replace_once(source, "from pathlib import Path\n", "from pathlib import Path\nimport os\n", "os import in Lightning module")
    source = replace_once(
        source,
        "        self.optim = optim\n",
        "        self.optim = optim\n"
        "        self.experiment_variant = os.environ.get('SHARP_EXPERIMENT_VARIANT', 'baseline')\n",
        "Lightning variant selection",
    )
    source = replace_once(
        source,
        "        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + new_agent_reg_loss + new_agent_cls_loss\n",
        "        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + new_agent_reg_loss + new_agent_cls_loss\n"
        "        if self.experiment_variant == 'endpoint_refinement_decoder':\n"
        "            from src.model.layers.ablation_modules import endpoint_auxiliary_loss\n"
        "            coarse_loss = endpoint_auxiliary_loss(\n"
        "                out['coarse_y_hat'], out['coarse_pi'], y\n"
        "            )\n"
        "            loss = loss + 0.2 * coarse_loss\n",
        "endpoint auxiliary loss",
    )
    source = replace_once(
        source,
        "            f'{tag}others_reg_loss': others_reg_loss.item(),\n",
        "            f'{tag}others_reg_loss': others_reg_loss.item(),\n"
        "            **({f'{tag}coarse_loss': coarse_loss.item()} if self.experiment_variant == 'endpoint_refinement_decoder' else {}),\n",
        "endpoint loss logging",
    )
    source = replace_once(
        source,
        "\n\nclass StreamLightningModule(BaseLightningModule):\n",
        '''\n\n    @staticmethod\n    def cross_window_consistency_loss(out, data):\n        previous = data.get("memory_new_y_hat")\n        memory = data.get("memory_dict")\n        if previous is None or memory is None or "pi" not in memory:\n            return out["y_hat"].sum() * 0.0\n        current = out["y_hat"][..., :2]\n        horizon = min(current.size(2), previous.size(2), 60)\n        current_probability = torch.softmax(out["pi"], dim=-1)\n        previous_probability = torch.softmax(memory["pi"].detach(), dim=-1)\n        current_mean = torch.sum(\n            current[:, :, :horizon] * current_probability.unsqueeze(-1).unsqueeze(-1),\n            dim=1,\n        )\n        previous_mean = torch.sum(\n            previous[:, :, :horizon].detach()\n            * previous_probability.unsqueeze(-1).unsqueeze(-1),\n            dim=1,\n        )\n        return F.smooth_l1_loss(current_mean, previous_mean)\n\n\nclass StreamLightningModule(BaseLightningModule):\n''',
        "cross-window consistency helper",
    )
    source = replace_once(
        source,
        "            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')\n            loss_dict.update(cur_loss_dict)\n",
        "            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')\n"
        "            if self.experiment_variant == 'cross_window_consistency':\n"
        "                consistency_loss = self.cross_window_consistency_loss(out, cur_data)\n"
        "                cur_loss = cur_loss + 0.05 * consistency_loss\n"
        "                cur_loss_dict[f'step{i + num_no_grad_frames}_consistency_loss'] = consistency_loss.item()\n"
        "            loss_dict.update(cur_loss_dict)\n",
        "cross-window consistency training loss",
    )
    source = replace_once(
        source,
        "        #assert len(param_dict.keys() - union_params) == 0\n",
        "        missing_parameters = param_dict.keys() - union_params\n"
        "        assert not missing_parameters, (\n"
        "            f'Trainable parameters missing from AdamW groups: {sorted(missing_parameters)}'\n"
        "        )\n",
        "optimizer coverage assertion",
    )
    write_lf(path, source)


def patch_train(code_dir: Path) -> None:
    path = code_dir / "train.py"
    source = path.read_text(encoding="utf-8")
    source = replace_once(source, "import os\n", "import os\nimport torch\n", "torch import")
    source = replace_once(
        source,
        "def main(cfg):\n",
        "def main(cfg):\n    torch.set_float32_matmul_precision('high')\n",
        "matmul precision",
    )
    source = replace_once(
        source,
        "    model = instantiate(cfg.model.pl_module)\n",
        '''    model = instantiate(cfg.model.pl_module)\n    variant = os.environ.get("SHARP_EXPERIMENT_VARIANT", "baseline")\n    marker_by_variant = {\n        "confidence_gated_memory": "memory_update_gate",\n        "learned_temporal_pool": "temporal_pool",\n        "uncertainty_target_context": "uncertainty_target_context",\n        "relative_geometry_bias": "relative_geometry_bias",\n        "kinematic_motion_stem": "kinematic_stem",\n        "endpoint_refinement_decoder": "endpoint_refinement",\n        "lane_topology_graph": "lane_graph_refiner",\n        "agent_temporal_mamba": "temporal_mamba",\n    }\n    parameter_names = [name for name, _ in model.named_parameters()]\n    expected_marker = marker_by_variant.get(variant)\n    if expected_marker and not any(expected_marker in name for name in parameter_names):\n        raise RuntimeError(f"Expected parameters for {variant} are missing")\n    if variant == "agent_temporal_mamba" and any("h_embed" in name for name in parameter_names):\n        raise RuntimeError("Temporal attention parameters remain in Mamba replacement")\n    if variant != "agent_temporal_mamba" and any("temporal_mamba" in name for name in parameter_names):\n        raise RuntimeError("Mamba parameters are active in a non-Mamba variant")\n    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)\n    print(f"SHARP_20EPOCH_VARIANT={variant} TRAINABLE_PARAMETERS={trainable}")\n    with open(os.path.join(output_dir, "VARIANT.txt"), "w") as handle:\n        handle.write(f"variant={variant}\\ntrainable_parameters={trainable}\\n")\n''',
        "runtime variant assertion",
    )
    source = replace_once(
        source,
        "    trainer.validate(model, datamodule.val_dataloader())\n",
        '''    best_path = trainer.checkpoint_callback.best_model_path\n    if not best_path:\n        raise RuntimeError("No best checkpoint was recorded")\n    print(f"FINAL_BEST_CHECKPOINT={best_path}")\n    trainer.validate(model, datamodule=datamodule, ckpt_path=best_path)\n''',
        "best-checkpoint validation",
    )
    write_lf(path, source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/server00/M")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    source_code = base / "Codes/SHARP/Code"
    dataset = base / "Datasets/AV2/sharp_processed"
    python_bin = base / "Codes/envs/sharp/bin/python"
    for required in (source_code, dataset / "train", dataset / "val"):
        if not required.is_dir():
            raise SystemExit(f"Missing required directory: {required}")
    if not python_bin.is_file():
        raise SystemExit(f"Missing Python environment: {python_bin}")

    original_sharp = (source_code / "src/model/sharp.py").read_text(encoding="utf-8")
    if "mamba" in original_sharp.lower():
        raise SystemExit("/home/server00/M/Codes/SHARP/Code is not clean original SHARP")

    mamba_packages = find_mamba_packages(base, python_bin)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_root = base / "Codes" / f"SHARP_AV2_20EPOCH_10TEST_{stamp}"
    results_root = base / "Results" / f"SHARP_AV2_20EPOCH_10TEST_{stamp}"
    code_dir = experiment_root / "Code"
    experiment_root.mkdir(parents=True, exist_ok=False)
    results_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source_code,
        code_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".hydra", "outputs"),
    )

    shutil.copy2(PACKAGE_ROOT / "runtime/ablation_modules.py", code_dir / "src/model/layers/ablation_modules.py")
    shutil.copy2(PACKAGE_ROOT / "runtime/temporal_mamba_replacement.py", code_dir / "src/model/layers/temporal_mamba_replacement.py")
    patch_common(code_dir)
    patch_sharp(code_dir)
    patch_lightning(code_dir)
    patch_train(code_dir)

    mamba_link = experiment_root / "mamba_packages"
    os.symlink(mamba_packages, mamba_link, target_is_directory=True)
    write_lf(experiment_root / "MAMBA_PACKAGE_SOURCE.txt", str(mamba_packages) + "\n")

    runtime_patch = experiment_root / "runtime_patch"
    runtime_patch.mkdir()
    shutil.copy2(PACKAGE_ROOT / "runtime/sitecustomize.py", runtime_patch / "sitecustomize.py")
    for filename in ("preflight_suite.py", "summarize_variant.py", "suite_status.py"):
        shutil.copy2(PACKAGE_ROOT / filename, experiment_root / filename)

    replacements = {
        "@@BASE@@": str(base),
        "@@EXPERIMENT_ROOT@@": str(experiment_root),
        "@@RESULTS_ROOT@@": str(results_root),
        "@@CODE_DIR@@": str(code_dir),
        "@@MAMBA_PACKAGES@@": str(mamba_link),
        "@@RUNTIME_PATCH@@": str(runtime_patch),
    }
    runner = (PACKAGE_ROOT / "run_10_test_suite.sh").read_text(encoding="utf-8")
    for key, value in replacements.items():
        runner = runner.replace(key, value)
    runner_path = experiment_root / "run_10_test_suite.sh"
    write_lf(runner_path, runner)
    make_executable(runner_path)

    manifest = (PACKAGE_ROOT / "EXPERIMENT_MANIFEST.md").read_text(encoding="utf-8")
    write_lf(experiment_root / "EXPERIMENT_MANIFEST.md", manifest)
    write_lf(base / "Codes/LATEST_SHARP_AV2_20EPOCH_10TEST.txt", str(experiment_root) + "\n")
    write_lf(base / "Results/LATEST_SHARP_AV2_20EPOCH_10TEST.txt", str(results_root) + "\n")

    print("SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"RUNNER={runner_path}")
    print("Previous code and results were not changed or deleted.")


if __name__ == "__main__":
    main()
