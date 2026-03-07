from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

_BATCH_NORM_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset_len: int, num_replicas: int, rank: int) -> None:
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= self.dataset_len:
            return 0
        return int(math.ceil((self.dataset_len - self.rank) / self.num_replicas))


def is_distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed_mode() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return DistributedContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=True,
            device=device,
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        local_rank = 0 if device.index is None else int(device.index)
    else:
        device = torch.device("cpu")
        local_rank = 0

    return DistributedContext(
        rank=0,
        local_rank=local_rank,
        world_size=1,
        is_distributed=False,
        device=device,
    )


def cleanup_distributed() -> None:
    if is_distributed_ready():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = getattr(model, "module", None)
    return module if module is not None else model


def count_unsynced_batchnorm_layers(model: torch.nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, _BATCH_NORM_TYPES) and not isinstance(module, torch.nn.SyncBatchNorm)
    )


def make_train_sampler(
    dataset,
    dist_ctx: DistributedContext,
    seed: int,
    drop_last: bool = True,
) -> DistributedSampler | None:
    if not dist_ctx.is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=dist_ctx.world_size,
        rank=dist_ctx.rank,
        shuffle=True,
        seed=int(seed),
        drop_last=drop_last,
    )


def make_eval_sampler(dataset, dist_ctx: DistributedContext) -> DistributedEvalSampler | None:
    if not dist_ctx.is_distributed:
        return None
    return DistributedEvalSampler(len(dataset), num_replicas=dist_ctx.world_size, rank=dist_ctx.rank)
