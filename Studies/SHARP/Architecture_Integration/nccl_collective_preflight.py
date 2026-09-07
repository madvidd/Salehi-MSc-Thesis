#!/usr/bin/env python3
"""Bounded four-GPU NCCL collective probe for the final Lab 2 suite."""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"Expected four ranks, got {world_size}")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=3),
        device_id=device,
    )
    try:
        for elements in (1, 768, 1_048_576):
            tensor = torch.full(
                (elements,), float(rank + 1), device=device, dtype=torch.float32
            )
            dist.broadcast(tensor, src=0)
            if not torch.all(tensor == 1):
                raise RuntimeError(f"Broadcast verification failed on rank {rank}")
            tensor.fill_(float(rank + 1))
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if not torch.all(tensor == 10):
                raise RuntimeError(f"All-reduce verification failed on rank {rank}")
        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize(device)
        print(f"NCCL_COLLECTIVE_RANK_OK={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
