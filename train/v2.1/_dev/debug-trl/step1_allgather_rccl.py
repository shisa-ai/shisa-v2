#!/usr/bin/env python3
"""Quick distributed stress test for large all-gather on MI300X (RCCL).

Launch with `torchrun --nproc_per_node=8 _dev/debug-trl/step1_allgather_rccl.py`.
The defaults mirror the 77,890,080 element shard observed in Qwen3-30B FSDP.
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large RCCL all-gather smoke test")
    parser.add_argument(
        "--numel",
        type=int,
        default=77_890_080,
        help="Number of elements per rank to all-gather (default matches Qwen3-30B shard)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Tensor dtype for the test payload",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="How many timed iterations to run after warmup",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before timing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="PRNG seed for reproducibility",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Distributed backend to use (default: nccl/RCCL)",
    )
    parser.add_argument(
        "--blocking-wait",
        action="store_true",
        help="Set TORCH_NCCL_BLOCKING_WAIT=1 for the duration of the test",
    )
    return parser.parse_args()


def dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name}") from exc


def setup_env(args: argparse.Namespace) -> None:
    if args.blocking_wait:
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_DEBUG", "INFO")
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", str(1 << 20))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    setup_env(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is required for this test")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=args.backend, device_id=local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dtype = dtype_from_string(args.dtype)
    numel = args.numel
    torch.manual_seed(args.seed + rank)

    element_bytes = torch.tensor([], dtype=dtype).element_size()
    shard_gib = numel * element_bytes / (1024 ** 3)
    total_gib = shard_gib * world_size

    if rank == 0:
        print(
            f"[info] world_size={world_size} dtype={dtype} numel_per_rank={numel} "
            f"shard_size={shard_gib:.3f} GiB aggregate={total_gib:.3f} GiB",
            flush=True,
        )

    payload = torch.randn(numel, dtype=dtype, device="cuda")
    gather_buf = torch.empty(numel * world_size, dtype=dtype, device="cuda")

    # Communicate that initialization succeeded
    dist.barrier()
    if rank == 0:
        print("[info] starting warmup", flush=True)

    # Warmup iterations to let RCCL pick algorithms
    for _ in range(args.warmup):
        dist.all_gather_into_tensor(gather_buf, payload)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("[info] timing all-gather iterations", flush=True)

    latencies = []
    for it in range(args.rounds):
        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()
        dist.all_gather_into_tensor(gather_buf, payload)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        latencies.append(elapsed)
        if rank == 0:
            bw = total_gib / elapsed if elapsed > 0 else float("inf")
            print(f"[iter {it}] latency={elapsed:.3f}s agg_bw={bw:.2f} GiB/s", flush=True)

    lat_tensor = torch.tensor(latencies, device="cuda")
    dist.reduce(lat_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        averaged = (lat_tensor / world_size).cpu()
        lat_mean = averaged.mean().item()
        lat_std = averaged.std().item() if args.rounds > 1 else float("nan")
        print(f"[summary] mean_latency={lat_mean:.3f}s std={lat_std:.3f}s over {args.rounds} rounds", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", flush=True)
        sys.exit(1)
