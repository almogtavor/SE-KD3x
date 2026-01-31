from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from sampledkd.config import TrainingConfig


@dataclass
class DistributedContext:
    """Lightweight container describing the distributed runtime."""

    world_size: int
    rank: int
    local_rank: int
    is_main_rank: bool
    distributed: bool


def is_rank0() -> bool:
    """Best-effort detection of global rank 0 before process group initialization."""
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except RuntimeError:
        pass
    rank = int(os.environ.get("RANK", "0") or 0)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    return rank == 0 or local_rank == 0


def setup_distributed_context(config: TrainingConfig) -> DistributedContext:
    """Initialize DDP (if requested) and persist context on the config."""
    ddp_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_env = ddp_world_size > 1
    ddp_rank = int(os.environ.get("RANK", "0")) if distributed_env else 0
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed_env else 0

    if distributed_env:
        torch.cuda.set_device(ddp_local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0
        distributed_env = False

    # Persist context back onto config for downstream components
    config.ddp_world_size = ddp_world_size
    config.ddp_rank = ddp_rank
    config.ddp_local_rank = ddp_local_rank

    return DistributedContext(
        world_size=ddp_world_size,
        rank=ddp_rank,
        local_rank=ddp_local_rank,
        is_main_rank=(ddp_rank == 0),
        distributed=distributed_env,
    )


def create_distributed_sampler(
    dataset,
    *,
    config: TrainingConfig,
    seed: int,
    shuffle: bool,
    drop_last: bool,
):
    if getattr(config, "ddp_world_size", 1) <= 1:
        return None
    return torch.utils.data.distributed.DistributedSampler(  # type: ignore[attr-defined]
        dataset,
        num_replicas=config.ddp_world_size,
        rank=config.ddp_rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def distributed_broadcast_object_list(obj_list, src: int = 0):
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(obj_list, src=src)
    return obj_list


def destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
