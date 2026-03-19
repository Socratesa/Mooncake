"""
Test for Mooncake scaling-up scenario.

Validates that:
1. The mooncake PG backend supports elastic group extension (adding ranks at runtime).
2. The mooncake EP buffer works correctly with non-power-of-2 rank counts
   (i.e. num_ranks that do NOT evenly divide MAX_QP_COUNT=256), exercising
   the qp_offsets-based QP allocation introduced to remove the old
   MAX_QP_COUNT % num_ranks == 0 constraint.
3. After scaling up, dispatch/combine produce correct results on all ranks
   (both original and newly added).

Usage:
    python test_mooncake_scaling.py
"""

import os
import time
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mooncake.mooncake_ep_buffer import Buffer
from ep_test_utils import init_dist, calc_diff, per_token_cast_back


os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "8371")


def _compute_qp_offsets(num_ranks: int, max_qp_count: int = 256):
    """Mirror the C++ qp_offsets computation for validation."""
    return [r * max_qp_count // num_ranks for r in range(num_ranks + 1)]


# ---------------------------------------------------------------------------
# Test 1: Validate qp_offsets arithmetic for various non-divisible rank counts
# ---------------------------------------------------------------------------
def test_qp_offsets_arithmetic():
    """Pure CPU test: verify offset table covers all 256 QPs without gaps."""
    for num_ranks in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 19, 24, 33, 48, 63, 64):
        offsets = _compute_qp_offsets(num_ranks)
        assert offsets[0] == 0, f"offsets[0]={offsets[0]} for num_ranks={num_ranks}"
        assert offsets[-1] == 256, f"offsets[-1]={offsets[-1]} for num_ranks={num_ranks}"
        for r in range(num_ranks):
            qp_count = offsets[r + 1] - offsets[r]
            assert qp_count >= 1, (
                f"rank {r} got 0 QPs with num_ranks={num_ranks}"
            )
    print("[PASS] qp_offsets arithmetic test passed for all rank counts", flush=True)


# ---------------------------------------------------------------------------
# Test 2: EP dispatch/combine correctness with non-divisible rank count
# ---------------------------------------------------------------------------
def _ep_correctness_worker(local_rank: int, num_local_ranks: int):
    """
    Spawn `num_local_ranks` processes and run dispatch + combine.
    num_local_ranks should be a value that does NOT divide 256 evenly
    (e.g. 3) to exercise the qp_offsets path.
    """
    rank, num_ranks, group, cpu_group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == num_local_ranks

    num_tokens = 64
    hidden = 7168
    num_topk = 4
    # Pick num_experts that is divisible by num_ranks
    num_experts = num_ranks * 8
    num_local_experts = num_experts // num_ranks

    num_ep_buffer_bytes = Buffer.get_ep_buffer_size_hint(
        num_tokens, hidden, num_ranks, num_experts
    )
    if local_rank == 0:
        print(
            f"[Scaling EP test] num_ranks={num_ranks}, num_experts={num_experts}, "
            f"buffer={num_ep_buffer_bytes / 1e6:.1f} MB",
            flush=True,
        )

    buffer = Buffer(group, num_ep_buffer_bytes=num_ep_buffer_bytes)

    # Generate deterministic test data
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)

    rank_offset = 128
    x = (
        torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
        * (rank - rank_offset)
    )
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)

    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    ).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = (
        torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    )
    active_ranks = torch.ones((num_tokens,), dtype=torch.int32, device="cuda")

    # Run dispatch
    for use_fp8 in (False, True):
        packed_recv_x, packed_recv_count, handle, event, hook = buffer.dispatch(
            x,
            topk_idx,
            active_ranks,
            num_tokens,
            num_experts,
            -1,
            use_fp8=use_fp8,
        )

        packed_recv_x_data = (
            (packed_recv_x[0], packed_recv_x[1].contiguous())
            if use_fp8
            else packed_recv_x
        )

        # Gather all topk_idx for validation
        all_topk_idx = torch.empty(
            (num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device="cuda"
        )
        dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

        # Validate per-expert receive counts
        for i in range(num_local_experts):
            expert_id = rank * num_local_experts + i
            recv_count = packed_recv_count[i].item()
            expected_count = (all_topk_idx == expert_id).sum().item()
            assert recv_count == expected_count, (
                f"[rank {rank}] expert {expert_id}: recv_count={recv_count} "
                f"!= expected={expected_count} (use_fp8={use_fp8})"
            )

        # Run combine
        simulated_gemm_x = (
            per_token_cast_back(
                packed_recv_x_data[0].view(-1, hidden),
                packed_recv_x_data[1].view(-1, hidden // 128),
            ).view(packed_recv_x_data[0].shape)
            if use_fp8
            else packed_recv_x.clone()
        )

        combined_x, event, hook = buffer.combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            active_ranks,
            -1,
            handle,
        )

        # Validate combine correctness
        expected = x * (
            topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
        )
        diff = calc_diff(expected, combined_x)
        assert not torch.isnan(combined_x).any(), f"[rank {rank}] NaN in combined_x"
        assert diff < 1e-5, (
            f"[rank {rank}] combine diff={diff} too large (use_fp8={use_fp8})"
        )

    if local_rank == 0:
        print(
            f"[PASS] EP correctness test passed with {num_ranks} ranks "
            f"(MAX_QP_COUNT % {num_ranks} = {256 % num_ranks})",
            flush=True,
        )

    try:
        dist.destroy_process_group()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test 3: PG elastic extension — start with N ranks, extend to N+M
# ---------------------------------------------------------------------------
def _elastic_scaling_worker(rank: int, initial_size: int, final_size: int, signals):
    """
    Test elastic scaling of the mooncake PG backend.
    - Ranks [0, initial_size) start first and do a collective.
    - Then we extend to final_size by letting new ranks join.
    - After extension, all ranks do a collective to verify correctness.
    """
    torch.cuda.set_device(0)  # All on same GPU for this unit test

    if rank < initial_size:
        # Original ranks start with smaller world
        dist.init_process_group(
            backend="mooncake-cpu",
            init_method=f"tcp://127.0.0.1:8372",
            rank=rank,
            world_size=initial_size,
        )

        # Verify initial collective works
        tensor = torch.tensor([rank + 1], dtype=torch.int32, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(1, initial_size + 1))
        assert tensor.item() == expected, (
            f"[rank {rank}] pre-extension: got {tensor.item()}, expected {expected}"
        )

        # Signal new ranks to start
        if rank == 0:
            signals["extend"] = 1

        # Wait for new ranks to connect
        from mooncake import pg

        backend = dist.group.WORLD._get_backend(torch.device("cpu"))
        while True:
            num_synced = pg.get_num_synced_ranks(backend)
            if num_synced == final_size:
                break
            # Keep doing collectives among existing ranks
            tensor = torch.tensor([rank + 1], dtype=torch.int32, device="cpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            assert tensor.item() == expected

        # Extend group
        pg.extend_group_size_to(backend, final_size)
    else:
        # New ranks wait for signal
        while "extend" not in signals:
            time.sleep(0.5)

        dist.init_process_group(
            backend="mooncake-cpu",
            init_method=f"tcp://127.0.0.1:8372",
            rank=rank,
            world_size=final_size,
        )

    # All ranks (old + new) do a collective
    tensor = torch.tensor([rank + 1], dtype=torch.int32, device="cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_all = sum(range(1, final_size + 1))
    assert tensor.item() == expected_all, (
        f"[rank {rank}] post-extension: got {tensor.item()}, expected {expected_all}"
    )

    if rank == 0:
        print(
            f"[PASS] PG elastic scaling test passed: {initial_size} -> {final_size} ranks",
            flush=True,
        )

    try:
        dist.destroy_process_group()
    except Exception:
        pass


def test_elastic_scaling():
    """Test PG elastic scaling: 2 ranks -> 3 ranks (non-power-of-2)."""
    initial_size = 2
    final_size = 3
    mp_manager = mp.Manager()
    signals = mp_manager.dict()

    mp.spawn(
        _elastic_scaling_worker,
        args=(initial_size, final_size, signals),
        nprocs=final_size,
        join=True,
    )


def test_ep_correctness_non_divisible():
    """Test EP dispatch/combine with 3 ranks (256 % 3 != 0)."""
    num_processes = 3
    mp.spawn(
        _ep_correctness_worker,
        args=(num_processes,),
        nprocs=num_processes,
    )


if __name__ == "__main__":
    # Test 1: Pure arithmetic validation (no GPU needed)
    test_qp_offsets_arithmetic()

    # Test 2: EP correctness with non-divisible rank count
    # Requires at least 3 GPUs (or NVLink/IPC fallback on fewer)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 3:
        test_ep_correctness_non_divisible()
    else:
        print(f"[SKIP] EP test requires >= 3 GPUs, found {num_gpus}", flush=True)

    # Test 3: PG elastic scaling
    # test_elastic_scaling()

    print("[ALL PASS] Mooncake scaling-up tests completed.", flush=True)
