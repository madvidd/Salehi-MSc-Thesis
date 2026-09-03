#!/usr/bin/env python3
import argparse
import gc
import math
import sys
from pathlib import Path

import torch


VARIANTS = (
    "baseline",
    "uncertainty_target_context",
    "relative_geometry_bias",
    "qknorm",
)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    code = Path(args.code).resolve()
    sys.path.insert(0, str(code))

    from src.datamodules.av2_datamodule import Av2DataModule
    from src.model.seam import Seam

    device = torch.device("cuda:0")
    datamodule = Av2DataModule(
        data_root=args.data_root,
        dataset={
            "num_historical_steps": 30,
            "split_points": [30, 40, 50],
            "radius": 150.0,
            "ma": False,
            "num_future_steps": 60,
        },
        train_batch_size=1,
        test_batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch = move_to_device(batch, device)
    if not isinstance(batch, list) or len(batch) < 2:
        raise RuntimeError("SEAM AV2 batch does not contain streamed observation windows")

    shared = dict(
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps=80,
        use_stream_encoder=True,
        use_stream_decoder=True,
        use_target_context=True,
        k=6,
        ma=False,
    )

    for variant in VARIANTS:
        torch.manual_seed(2333)
        torch.cuda.manual_seed_all(2333)
        model = Seam(variant=variant, **shared).to(device).train()
        memory = None
        loss = None
        for index, frame in enumerate(batch):
            frame = dict(frame)
            frame["memory_dict"] = memory
            if index < max(0, len(batch) - 3):
                with torch.no_grad():
                    output = model(frame)
            else:
                output = model(frame)
                term = output["y_hat"].float().square().mean()
                term = term + output["pi"].float().square().mean()
                loss = term if loss is None else loss + term
            memory = output["memory_dict"]

        if loss is None or not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError(f"Non-finite smoke-test loss for {variant}")
        loss.backward()

        required_gradients = {
            "uncertainty_target_context": "uncertainty_target_context.controller.weight",
            "relative_geometry_bias": "relative_geometry_bias.network.2.weight",
            "qknorm": "h_embed.0.attn.logit_scale",
        }
        required = required_gradients.get(variant)
        if required:
            parameter = dict(model.named_parameters()).get(required)
            if parameter is None or parameter.grad is None:
                raise RuntimeError(
                    f"The intervention parameter {required} was unused for {variant}"
                )

        print(
            f"REAL_DATA_CUDA_SMOKE_OK={variant};"
            f"windows={len(batch)};loss={float(loss.detach().cpu()):.6f}"
        )
        del model, output, memory, loss
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    print("ALL_REAL_DATA_SMOKE_TESTS_OK=True")


if __name__ == "__main__":
    main()
