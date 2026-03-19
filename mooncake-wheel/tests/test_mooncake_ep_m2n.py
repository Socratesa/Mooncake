"""
M2N (Attention-FFN Disaggregation) Tests for Mooncake EP.

Test 1: Basic M2N Dispatch + Combine
Test 2: Scaling Down (rank failure via active_ranks) + numeric correctness
Test 3a: Scaling Up — Recovery (update_ep_member, no rebuild)
Test 3b: Scaling Up — Expansion via scale_up() (4->8 rank, 2a+6f with replication)
Test 4: CUDA Graph steady-state capture/replay

NOTE: The logical-to-physical expert mapping (build_phy2log, ExpertRemapper)
is inference engine responsibility (e.g. SGLang EPLB). These helpers are
included here to simulate what the inference engine would do. Mooncake EP
only sees physical expert IDs.

active_ranks is owned by the inference engine and passed to every
dispatch/combine call. This is required for CUDA graph compatibility.
"""

import sys
import time
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple

from mooncake.mooncake_ep_m2n_buffer import M2NBuffer
from ep_test_utils import init_dist, calc_diff, per_token_cast_back


def nccl_barrier(group):
    """NCCL-compatible barrier using a dummy all_reduce."""
    t = torch.zeros(1, device="cuda")
    dist.all_reduce(t, group=group)
    torch.cuda.synchronize()


# ===========================================================================
# Inference engine helpers (would live in SGLang / inference engine side)
# ===========================================================================

def build_phy2log(
    num_ranks: int,
    attention_ranks: List[int],
    ffn_ranks: List[int],
    num_logical_experts: int,
    ffn_expert_assignment: Optional[Dict[int, List[int]]] = None,
) -> List[int]:
    """
    Build phy2log mapping for M2N configuration.

    This is inference engine logic -- Mooncake EP does not need this.

    Args:
        num_ranks: total number of ranks.
        attention_ranks: list of attention rank ids.
        ffn_ranks: list of FFN rank ids.
        num_logical_experts: number of model logical experts.
        ffn_expert_assignment: optional dict mapping ffn_rank -> list of logical expert ids.

    Returns:
        phy2log: list of length num_experts_total, where
                 phy2log[physical_id] = logical expert id (-1 for virtual slots).
    """
    num_ffn = len(ffn_ranks)

    if ffn_expert_assignment is None:
        assert num_logical_experts % num_ffn == 0
        num_experts_per_rank = num_logical_experts // num_ffn
    else:
        max_assigned = max(len(v) for v in ffn_expert_assignment.values())
        num_experts_per_rank = max_assigned

    num_experts_total = num_ranks * num_experts_per_rank

    phy2log = [-1] * num_experts_total

    if ffn_expert_assignment is None:
        for i, ffn_rank in enumerate(sorted(ffn_ranks)):
            base_logical = i * num_experts_per_rank
            base_physical = ffn_rank * num_experts_per_rank
            for j in range(num_experts_per_rank):
                phy2log[base_physical + j] = base_logical + j
    else:
        for ffn_rank, logical_ids in ffn_expert_assignment.items():
            assert ffn_rank in ffn_ranks
            base_physical = ffn_rank * num_experts_per_rank
            for j, log_id in enumerate(logical_ids):
                phy2log[base_physical + j] = log_id

    return phy2log


class ExpertRemapper:
    """
    Inference engine side: maps logical expert IDs to physical slot IDs.

    This would live in SGLang's EPLB or router module. Mooncake EP never
    sees logical expert IDs -- only physical slot IDs after this remapping.
    """

    def __init__(self, phy2log: List[int], num_logical_experts: int):
        self.num_logical_experts = num_logical_experts

        # Build log2phy: for each logical expert, collect all physical replicas
        log2phy_lists: List[List[int]] = [[] for _ in range(num_logical_experts)]
        for phys_id, log_id in enumerate(phy2log):
            if 0 <= log_id < num_logical_experts:
                log2phy_lists[log_id].append(phys_id)

        for log_id in range(num_logical_experts):
            assert len(log2phy_lists[log_id]) > 0, \
                f"logical expert {log_id} has no physical slot"

        # GPU lookup table (first replica -- fast path for no replication)
        self._log2phy_first = torch.tensor(
            [slots[0] for slots in log2phy_lists],
            dtype=torch.int64, device='cuda',
        )

        # Padded table for random replica selection
        max_replicas = max(len(slots) for slots in log2phy_lists)
        self._max_replicas = max_replicas
        self._log2phy_padded = torch.full(
            (num_logical_experts, max_replicas), -1, dtype=torch.int64, device='cuda',
        )
        self._replica_counts = torch.zeros(
            num_logical_experts, dtype=torch.int64, device='cuda',
        )
        for log_id, slots in enumerate(log2phy_lists):
            self._replica_counts[log_id] = len(slots)
            for j, phys_id in enumerate(slots):
                self._log2phy_padded[log_id, j] = phys_id

    def remap(self, topk_idx_logical: torch.Tensor) -> torch.Tensor:
        """
        Remap logical expert IDs to physical slot IDs.

        When replication exists, randomly selects a replica per element.

        Args:
            topk_idx_logical: [num_tokens, num_topk] int64, logical expert IDs.
                              -1 means invalid/padding.

        Returns:
            topk_idx_physical: [num_tokens, num_topk] int64, physical slot IDs.
        """
        mask = topk_idx_logical >= 0
        safe_idx = topk_idx_logical.clamp(min=0)
        if self._max_replicas == 1:
            topk_idx_physical = self._log2phy_first[safe_idx]
        else:
            counts = self._replica_counts[safe_idx]
            rand_idx = (torch.rand(topk_idx_logical.shape, device='cuda') * counts.float()).long()
            rand_idx = rand_idx.clamp(max=self._max_replicas - 1)
            topk_idx_physical = self._log2phy_padded[safe_idx, rand_idx]
        topk_idx_physical = topk_idx_physical.masked_fill(~mask, -1)
        return topk_idx_physical


# ===========================================================================
# Test helpers
# ===========================================================================

def log(rank: int, role: str, msg: str):
    role_tag = "Attn" if role == "attention" else "FFN"
    print(f"[Rank {rank} ({role_tag})] {msg}", flush=True)


def make_tokens(num_tokens: int, hidden: int, rank: int):
    """Create distinguishable token data."""
    x = torch.full((num_tokens, hidden), float(rank), dtype=torch.bfloat16, device="cuda")
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    return x


def make_routing(num_tokens: int, num_logical_experts: int, num_topk: int, seed: int):
    """Generate random routing decisions (logical expert IDs)."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    scores = torch.rand(num_tokens, num_logical_experts, generator=gen, device="cuda")
    topk_idx = scores.topk(num_topk, dim=-1).indices.to(torch.int64)
    topk_weights = torch.rand(num_tokens, num_topk, generator=gen, device="cuda", dtype=torch.float32).abs() + 0.1
    return topk_idx, topk_weights


def make_active_ranks(num_ranks: int) -> torch.Tensor:
    """Create active_ranks tensor (inference engine owns this)."""
    return torch.ones(num_ranks, dtype=torch.int32, device="cuda")


def simulate_expert_fn(packed_recv_x, packed_recv_count, num_experts_per_rank, use_fp8: bool):
    """Simulate expert computation: identity (output = input)."""
    if use_fp8:
        x_fp8, x_scales = packed_recv_x
        expert_out = per_token_cast_back(
            x_fp8.view(-1, x_fp8.size(-1)),
            x_scales.view(-1, x_scales.size(-1)),
        ).view(x_fp8.shape).to(torch.bfloat16)
    else:
        expert_out = packed_recv_x.clone()
    return expert_out


def print_routing_stats(rank: int, role: str, topk_idx_physical: torch.Tensor,
                        num_ranks: int, num_experts_per_rank: int, label: str = "routing"):
    """Print per-rank dispatch token count distribution."""
    tokens_per_rank = []
    for r in range(num_ranks):
        base = r * num_experts_per_rank
        cnt = ((topk_idx_physical >= base) & (topk_idx_physical < base + num_experts_per_rank)).sum().item()
        tokens_per_rank.append(int(cnt))
    valid_total = (topk_idx_physical >= 0).sum().item()
    log(rank, role, f"[{label}] topk entries per rank: {tokens_per_rank}  (valid total={int(valid_total)})")


def run_dispatch_combine(
    m2n: M2NBuffer,
    active_ranks: torch.Tensor,
    x: torch.Tensor,
    topk_idx_physical: torch.Tensor,
    topk_weights: torch.Tensor,
    num_topk: int,
    timeout_us: int = -1,
    use_fp8: bool = False,
    return_recv_hook: bool = False,
):
    """
    Run a full dispatch + combine cycle.

    active_ranks and topk_idx_physical are managed by the inference engine.
    When return_recv_hook=True, dispatch/combine are split into send + recv
    phases to allow overlapping compute between them.
    Returns (combined_x, recv_x, packed_recv_count).
    """
    t0 = time.monotonic()

    # --- Dispatch phase ---
    if m2n.role == "attention":
        recv_x, packed_recv_count, handle, event, hook = m2n.a2e_isend(
            x, topk_idx_physical, active_ranks,
            timeout_us=timeout_us, use_fp8=use_fp8,
            return_recv_hook=return_recv_hook,
        )
    else:
        recv_x, packed_recv_count, handle, event, hook = m2n.a2e_irecv(
            active_ranks, num_topk, timeout_us=timeout_us, use_fp8=use_fp8,
            return_recv_hook=return_recv_hook,
        )
    if hook is not None:
        # Overlap window: user compute can go here before hook() waits for RDMA.
        # recv_x / packed_recv_count are NOT ready yet at this point.
        log(m2n.rank, m2n.role, "dispatch issued — overlap compute window open")
        hook()
        torch.cuda.synchronize()
        # Data is now fully ready after hook().
        if m2n.role == "ffn":
            total_recv = packed_recv_count.sum().item()
            nonzero_experts = (packed_recv_count > 0).sum().item()
            log(m2n.rank, m2n.role,
                f"[dispatch done] recv_x shape={tuple(recv_x.shape)}, "
                f"packed_recv_count total={int(total_recv)} tokens "
                f"across {nonzero_experts}/{packed_recv_count.numel()} experts, "
                f"counts={packed_recv_count.tolist()}")
        else:
            # For attention: recv_x is the pre-allocated combine output buffer,
            # not dispatch data. It will be filled by FFN during combine phase.
            log(m2n.rank, m2n.role,
                f"[dispatch done] combine output buf pre-allocated, shape={tuple(recv_x.shape)}")

    t1 = time.monotonic()

    # --- Combine phase ---
    if m2n.role == "ffn":
        expert_out = simulate_expert_fn(recv_x, packed_recv_count, m2n.num_experts_per_rank, use_fp8)
        combined_x, event, hook = m2n.e2a_isend(
            expert_out, active_ranks, handle, timeout_us=timeout_us,
            return_recv_hook=return_recv_hook,
        )
    else:
        combined_x, event, hook = m2n.e2a_irecv(
            topk_idx_physical, topk_weights, active_ranks, handle,
            timeout_us=timeout_us,
            return_recv_hook=return_recv_hook,
        )
    if hook is not None:
        log(m2n.rank, m2n.role, "combine issued — overlap compute window open")
        hook()
        torch.cuda.synchronize()
        if m2n.role == "attention":
            has_nan = torch.isnan(combined_x).sum().item()
            log(m2n.rank, m2n.role,
                f"[combine done] combined_x shape={tuple(combined_x.shape)}, "
                f"nan_count={has_nan}, sample[0]={combined_x[0, :4].tolist()} ...")
        else:
            log(m2n.rank, m2n.role,
                f"[combine done] combined_x shape={tuple(combined_x.shape)} (FFN sends back, no local result)")

    t2 = time.monotonic()
    hook_tag = " [hook]" if return_recv_hook else ""
    print(f"  [{m2n.role}] dispatch={1000*(t1-t0):.1f}ms, combine={1000*(t2-t1):.1f}ms, "
          f"total={1000*(t2-t0):.1f}ms{hook_tag}", flush=True)

    return combined_x, recv_x, packed_recv_count


def verify_combine_correctness(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    combined_x: torch.Tensor,
    label: str,
    rank: int,
    role: str,
):
    """Verify combined_x matches expected weighted sum (identity expert)."""
    if role != "attention":
        return
    valid_weights = topk_weights.masked_fill(topk_idx < 0, 0.0)
    expected = x * valid_weights.sum(dim=1, keepdim=True)
    diff = calc_diff(expected, combined_x)
    has_nan = torch.isnan(combined_x).sum().item()
    log(rank, role, f"{label}: diff = {diff:.2e}, nan_count = {has_nan}")
    assert has_nan == 0, f"{label}: combined_x has NaN"
    assert diff < 1e-3, f"{label}: diff {diff} too large"


def verify_combine_with_active_mask(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    combined_x: torch.Tensor,
    active_ranks: torch.Tensor,
    num_experts_per_rank: int,
    label: str,
    rank: int,
    role: str,
):
    """
    Verify combined_x when some ranks are inactive.

    Tokens routed fully to active ranks must match expected correctness.
    Mixed tokens (some experts on inactive ranks) may contain stale data,
    so we only check they are NaN-free.
    """
    if role != "attention":
        return

    has_nan = torch.isnan(combined_x).sum().item()
    assert has_nan == 0, f"{label}: combined_x has NaN"

    # Build a mask of which expert slots belong to active ranks
    num_ranks = active_ranks.shape[0]
    num_experts_total = num_ranks * num_experts_per_rank
    active_expert_mask = torch.zeros(num_experts_total, dtype=torch.bool, device="cuda")
    for r in range(num_ranks):
        if active_ranks[r].item() == 1:
            base = r * num_experts_per_rank
            active_expert_mask[base:base + num_experts_per_rank] = True

    # Identify tokens routed ONLY to active ranks
    topk_flat = topk_idx.view(-1)
    safe_idx = topk_flat.clamp(min=0)
    is_valid = topk_flat >= 0
    is_active_slot = active_expert_mask[safe_idx] & is_valid
    inactive_slot_hits = (~is_active_slot & is_valid).view(topk_idx.shape)
    tokens_all_active = ~inactive_slot_hits.any(dim=1)
    num_all_active = tokens_all_active.sum().item()
    log(rank, role, f"{label}: {num_all_active}/{x.shape[0]} tokens fully active")

    # Tokens fully on active ranks must be numerically correct
    if num_all_active > 0:
        valid_weights_all = topk_weights.masked_fill(topk_idx < 0, 0.0)
        expected_full = x * valid_weights_all.sum(dim=1, keepdim=True)
        diff_full = calc_diff(
            expected_full[tokens_all_active],
            combined_x[tokens_all_active],
        )
        log(rank, role, f"{label}: fully-active tokens diff = {diff_full:.2e}")
        assert diff_full < 1e-3, f"{label}: fully-active diff {diff_full} too large"


def mask_inactive_experts(
    topk_idx: torch.Tensor,
    active_ranks: torch.Tensor,
    num_experts_per_rank: int,
) -> torch.Tensor:
    """Mask topk_idx entries targeting inactive rank expert slots to -1.

    Simulates what the inference engine does after discovering a rank is
    down via active_ranks: re-route tokens away from the failed rank by
    setting those expert entries to padding (-1).
    """
    num_ranks = active_ranks.shape[0]
    num_experts_total = num_ranks * num_experts_per_rank
    active_expert_mask = torch.zeros(num_experts_total, dtype=torch.bool, device="cuda")
    for r in range(num_ranks):
        if active_ranks[r].item() == 1:
            base = r * num_experts_per_rank
            active_expert_mask[base:base + num_experts_per_rank] = True

    topk_idx_masked = topk_idx.clone()
    flat = topk_idx_masked.view(-1)
    is_valid = flat >= 0
    safe_idx = flat.clamp(min=0)
    is_inactive = is_valid & ~active_expert_mask[safe_idx]
    flat[is_inactive] = -1
    return topk_idx_masked


# ===========================================================================
# Test 1: Basic M2N Dispatch + Combine
# ===========================================================================

def test_basic_m2n(rank: int, num_ranks: int, group: dist.ProcessGroup):
    """Test basic M2N dispatch + combine with 2 attention + 2 FFN ranks."""
    num_logical_experts = 128
    hidden = 2560
    num_topk = 8
    num_tokens = 128
    num_max_dispatch_tokens_per_rank = 128

    attention_ranks = [0, 1]
    ffn_ranks = [2, 3]

    # --- Inference engine side: build mapping ---
    phy2log = build_phy2log(num_ranks, attention_ranks, ffn_ranks, num_logical_experts)
    num_experts_per_rank = len(phy2log) // num_ranks
    remapper = ExpertRemapper(phy2log, num_logical_experts)
    active_ranks = make_active_ranks(num_ranks)

    # --- Mooncake side: create buffer (no mapping knowledge) ---
    m2n = M2NBuffer(
        group, attention_ranks, ffn_ranks, num_experts_per_rank,
        num_max_dispatch_tokens_per_rank, hidden,
    )
    role = m2n.role

    log(rank, role, f"M2NBuffer created: num_experts_total={m2n.num_experts_total}, "
                     f"num_experts_per_rank={m2n.num_experts_per_rank}, "
                     f"attention_ranks={attention_ranks}, ffn_ranks={ffn_ranks}")

    x = make_tokens(num_tokens, hidden, rank) if role == "attention" else torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx_logical, topk_weights = make_routing(num_tokens, num_logical_experts, num_topk, seed=42 + rank)

    # --- Inference engine side: remap logical -> physical ---
    topk_idx_physical = remapper.remap(topk_idx_logical)

    if role == "attention":
        log(rank, role,
            f"input x: shape={tuple(x.shape)}, dtype={x.dtype}, "
            f"fill_value={float(rank):.0f}, sample x[0,:4]={x[0, :4].tolist()}")
        print_routing_stats(rank, role, topk_idx_physical, num_ranks, num_experts_per_rank, "dispatch routing")

    # --- Round 1: non-hook mode ---
    log(rank, role, "---- Round 1: non-hook mode ----")
    combined_x, recv_x, packed_recv_count = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk, use_fp8=False,
    )

    if role == "ffn":
        total_recv = packed_recv_count.sum().item()
        log(rank, role,
            f"dispatch recv: total {int(total_recv)} tokens across "
            f"{m2n.num_experts_per_rank} local experts, "
            f"counts={packed_recv_count.tolist()}")

    if role == "attention":
        log(rank, role,
            f"combined_x: shape={tuple(combined_x.shape)}, "
            f"sample[0]={combined_x[0, :4].tolist()} ...")

    verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x, "Test 1 (basic)", rank, role)

    nccl_barrier(group)

    # --- Round 2: hook mode (overlap compute between dispatch and combine) ---
    log(rank, role, "---- Round 2: return_recv_hook=True (overlap mode) ----")
    combined_x_hook, _, _ = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk,
        use_fp8=False, return_recv_hook=True,
    )

    verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x_hook,
                               "Test 1 (return_recv_hook)", rank, role)

    nccl_barrier(group)
    if rank == 0:
        print("=" * 60)
        print("Test 1: Basic M2N Dispatch + Combine -- PASSED")
        print("=" * 60, flush=True)


# ===========================================================================
# Test 2: Scaling Down (active_ranks managed by inference engine)
# ===========================================================================

def test_scaling_down(rank: int, num_ranks: int, group: dist.ProcessGroup):
    """Test scaling down by simulating rank 3 (FFN) failure.

    Verifies realistic failure → detection → re-route flow:

    Phase 1: All 4 ranks participate — baseline correctness.
    Phase 2a (simulate crash): Rank 3 **skips** dispatch/combine entirely
        (simulating process crash). Other ranks call dispatch/combine with
        timeout_us. The RDMA kernel spins waiting for rank 3's signal,
        times out, and sets active_ranks[3]=0 **at the kernel level** —
        this is runtime detection, not a test preset.
        Verification: NaN-free, tokens routed only to surviving ranks
        match expected values.
    Phase 2b (re-route): Inference engine inspects active_ranks (now
        reflecting kernel-discovered failure), masks topk_idx entries for
        inactive rank experts to -1. Full numeric correctness for all
        tokens.
    """
    num_logical_experts = 128
    hidden = 2560
    num_topk = 8
    num_tokens = 128
    num_max_dispatch_tokens_per_rank = 128

    attention_ranks = [0, 1]
    ffn_ranks = [2, 3]

    # --- Inference engine side ---
    phy2log = build_phy2log(num_ranks, attention_ranks, ffn_ranks, num_logical_experts)
    num_experts_per_rank = len(phy2log) // num_ranks
    remapper = ExpertRemapper(phy2log, num_logical_experts)
    active_ranks = make_active_ranks(num_ranks)

    # --- Mooncake side ---
    m2n = M2NBuffer(
        group, attention_ranks, ffn_ranks, num_experts_per_rank,
        num_max_dispatch_tokens_per_rank, hidden,
    )
    role = m2n.role

    x = make_tokens(num_tokens, hidden, rank) if role == "attention" else torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx_logical, topk_weights = make_routing(num_tokens, num_logical_experts, num_topk, seed=100 + rank)
    topk_idx_physical = remapper.remap(topk_idx_logical)

    if role == "attention":
        print_routing_stats(rank, role, topk_idx_physical, num_ranks, num_experts_per_rank, "phase1 routing")

    # Phase 1: normal dispatch + combine (all 4 ranks)
    log(rank, role, f"---- Phase 1: all ranks active, active_ranks={active_ranks.tolist()} ----")
    combined_x, _, _ = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk, use_fp8=False,
    )
    if role == "attention":
        verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x, "Test 2 Phase 1 (normal)", rank, role)

    nccl_barrier(group)

    # Phase 2a: rank 3 "crashes" — does not call dispatch/combine.
    # Other ranks proceed with timeout; kernel sets active_ranks[3]=0
    # upon detecting the missing signal from rank 3.
    if rank == 3:
        # Simulate crash: rank 3 does nothing during this round.
        log(rank, role, "---- Phase 2a: SIMULATING CRASH — skipping dispatch/combine ----")
        torch.cuda.synchronize()
    else:
        log(rank, role,
            f"---- Phase 2a: rank 3 crash simulation, timeout=1s, "
            f"active_ranks={active_ranks.tolist()} ----")
        combined_x2a, _, _ = run_dispatch_combine(
            m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk,
            timeout_us=1_000_000, use_fp8=False,
        )

        # After the kernel returns, active_ranks[3] should be 0
        # (set by the kernel's timeout logic, not by us)
        rank3_status = active_ranks[3].item()
        log(rank, role,
            f"Phase 2a done: kernel set active_ranks[3]={rank3_status}, "
            f"active_ranks now={active_ranks.tolist()}")
        assert rank3_status == 0, (
            f"Expected kernel to set active_ranks[3]=0 via timeout, "
            f"but got {rank3_status}"
        )

        verify_combine_with_active_mask(
            x, topk_idx_physical, topk_weights, combined_x2a, active_ranks,
            num_experts_per_rank, "Test 2 Phase 2a (rank 3 crashed, kernel detected)", rank, role,
        )

    # Sync: rank 3 must still participate in barrier (process is alive,
    # only simulating crash at the dispatch/combine level)
    nccl_barrier(group)

    # Phase 2b: inference engine discovers active_ranks[3]=0, re-routes
    # by masking topk_idx for inactive rank expert slots to -1.
    # All 4 ranks participate (rank 3 is "recovering" but still sends
    # empty contributions with active_ranks[3]=0).
    if rank != 3:
        topk_idx_masked = mask_inactive_experts(topk_idx_physical, active_ranks, num_experts_per_rank)
    else:
        # Rank 3 also needs the failure state propagated
        active_ranks[3] = 0
        topk_idx_masked = mask_inactive_experts(topk_idx_physical, active_ranks, num_experts_per_rank)

    num_masked = (topk_idx_masked == -1).sum().item() - (topk_idx_physical == -1).sum().item()
    log(rank, role,
        f"---- Phase 2b: re-route, {num_masked} entries masked to -1 "
        f"(rank3 physical slots [{3*num_experts_per_rank}, {4*num_experts_per_rank})), "
        f"active_ranks={active_ranks.tolist()} ----")

    if role == "attention":
        print_routing_stats(rank, role, topk_idx_masked, num_ranks, num_experts_per_rank, "phase2b routing (masked)")

    combined_x2b, _, _ = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_masked, topk_weights, num_topk,
        timeout_us=1_000_000, use_fp8=False,
    )

    if role == "attention":
        verify_combine_correctness(x, topk_idx_masked, topk_weights, combined_x2b,
                                   "Test 2 Phase 2b (re-routed)", rank, role)

    nccl_barrier(group)
    if rank == 0:
        print("=" * 60)
        print("Test 2: Scaling Down -- PASSED")
        print("=" * 60, flush=True)


# ===========================================================================
# Test 3a: Scaling Up -- Recovery (no rebuild, use update_ep_member)
# ===========================================================================

def test_scaling_up_recovery(rank: int, num_ranks: int, group: dist.ProcessGroup):
    """Test rank failure followed by recovery via update_ep_member.

    Phase 1: All 4 ranks active — baseline correctness.
    Phase 2: Rank 3 is completely absent (does not call dispatch/combine at all).
        Inference engine pre-sets active_ranks[3]=0 AND masks topk_idx entries
        for rank 3 experts to -1 BEFORE dispatch. This means:
          - No tokens are sent to rank 3 (topk_idx_masked)
          - The combine kernel skips rank 3 slots immediately (active_ranks[3]=0)
          - No timeout needed; results are fully numerically correct and verified.
        This is a genuine absence test: rank 3's buffer_idx does NOT advance,
        diverging from the other ranks.
    Phase 3: All ranks call update_ep_member() to re-synchronise RDMA state
        (including resetting rank 3's diverged buffer index), then active_ranks[3]=1
        is restored. Full numeric correctness is verified, confirming update_ep_member
        truly re-establishes communication with rank 3.
    """
    num_logical_experts = 128
    hidden = 2560
    num_topk = 8
    num_tokens = 128
    num_max_dispatch_tokens_per_rank = 128

    attention_ranks = [0, 1]
    ffn_ranks = [2, 3]

    # --- Inference engine side ---
    phy2log = build_phy2log(num_ranks, attention_ranks, ffn_ranks, num_logical_experts)
    num_experts_per_rank = len(phy2log) // num_ranks
    remapper = ExpertRemapper(phy2log, num_logical_experts)
    active_ranks = make_active_ranks(num_ranks)

    # --- Mooncake side ---
    m2n = M2NBuffer(
        group, attention_ranks, ffn_ranks, num_experts_per_rank,
        num_max_dispatch_tokens_per_rank, hidden,
    )
    role = m2n.role

    x = make_tokens(num_tokens, hidden, rank) if role == "attention" else torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx_logical, topk_weights = make_routing(num_tokens, num_logical_experts, num_topk, seed=200 + rank)
    topk_idx_physical = remapper.remap(topk_idx_logical)

    # Phase 1: normal baseline
    log(rank, role, f"---- Phase 1: all ranks active, active_ranks={active_ranks.tolist()} ----")
    if role == "attention":
        print_routing_stats(rank, role, topk_idx_physical, num_ranks, num_experts_per_rank, "phase1 routing")
    combined_x, _, _ = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk, use_fp8=False,
    )
    if role == "attention":
        verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x, "Test 3a Phase 1 (normal)", rank, role)
    nccl_barrier(group)

    # Phase 2: rank 3 genuinely absent — does not call dispatch/combine at all.
    # Inference engine pre-sets active_ranks[3]=0 and masks topk_idx so that:
    #   (a) no tokens are dispatched to rank 3 experts (topk_idx_masked)
    #   (b) the combine kernel exits rank-3 spin-wait immediately without timeout
    # The remaining ranks produce fully correct results with no timeout penalty.
    active_ranks[3] = 0
    topk_idx_masked = mask_inactive_experts(topk_idx_physical, active_ranks, num_experts_per_rank)
    num_masked = (topk_idx_masked == -1).sum().item() - (topk_idx_physical == -1).sum().item()
    log(rank, role,
        f"---- Phase 2: rank 3 offline — {num_masked} topk entries masked, "
        f"active_ranks={active_ranks.tolist()} ----")

    if rank == 3:
        # Rank 3 completely absent: buffer_idx intentionally NOT advanced here.
        # update_ep_member() in Phase 3 must resync this diverged state.
        log(rank, role, "Phase 2: rank 3 offline — skipping dispatch/combine entirely")
        torch.cuda.synchronize()
    else:
        if role == "attention":
            print_routing_stats(rank, role, topk_idx_masked, num_ranks, num_experts_per_rank, "phase2 routing (masked)")
        combined_x2, _, _ = run_dispatch_combine(
            m2n, active_ranks, x, topk_idx_masked, topk_weights, num_topk, use_fp8=False,
        )
        if role == "attention":
            # Full numeric verify: all tokens are masked away from rank 3,
            # so combined_x is cleanly rank-2-only, diff must be tight.
            verify_combine_correctness(x, topk_idx_masked, topk_weights, combined_x2,
                                       "Test 3a Phase 2 (rank 3 absent)", rank, role)
    nccl_barrier(group)

    # Phase 3: recover — update_ep_member resyncs RDMA state (including rank 3's
    # diverged buffer_idx), then re-enable rank 3 with original routing.
    log(rank, role, f"---- Phase 3: recovery — update_ep_member + active_ranks[3] = 1 ----")
    m2n.update_ep_member()
    active_ranks[3] = 1
    log(rank, role, f"active_ranks restored to: {active_ranks.tolist()}")

    if role == "attention":
        print_routing_stats(rank, role, topk_idx_physical, num_ranks, num_experts_per_rank, "phase3 routing (restored)")
    combined_x3, _, _ = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk, use_fp8=False,
    )
    if role == "attention":
        verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x3,
                                   "Test 3a Phase 3 (recovered)", rank, role)

    nccl_barrier(group)
    if rank == 0:
        print("=" * 60)
        print("Test 3a: Scaling Up -- Recovery -- PASSED")
        print("=" * 60, flush=True)


# ===========================================================================
# Test 3b: Scaling Up -- Expansion via scale_up() (4->8 rank, 2a+6f)
# ===========================================================================

def test_scaling_up_expansion(rank: int, num_ranks: int, group: dist.ProcessGroup):
    """
    Test expansion from 4 rank (2a+2f) to 8 rank (2a+6f).
    Phase 2 uses scale_up() on existing M2NBuffer (ranks 0-3) instead of
    constructing a new M2NBuffer, exercising the real rebuild path.
    Requires 8 GPUs.
    """
    num_logical_experts = 128
    hidden = 2560
    num_topk = 8
    num_tokens = 128
    num_max_dispatch_tokens_per_rank = 128

    assert num_ranks == 8, "Test 3b requires 8 ranks"

    # --- Phase 1: 4 ranks (2a+2f) ---
    phase1_ranks = list(range(4))
    phase1_attention = [0, 1]
    phase1_ffn = [2, 3]

    m2n = None
    if rank < 4:
        phase1_group = dist.new_group(phase1_ranks)
        phase1_active_ranks = make_active_ranks(4)

        # Inference engine side
        phase1_phy2log = build_phy2log(4, phase1_attention, phase1_ffn, num_logical_experts)
        phase1_num_experts_per_rank = len(phase1_phy2log) // 4
        phase1_remapper = ExpertRemapper(phase1_phy2log, num_logical_experts)

        # Mooncake side
        m2n = M2NBuffer(
            phase1_group, phase1_attention, phase1_ffn,
            phase1_num_experts_per_rank, num_max_dispatch_tokens_per_rank, hidden,
        )
        phase1_role = m2n.role

        x = make_tokens(num_tokens, hidden, rank) if phase1_role == "attention" else torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
        topk_idx_logical, topk_weights = make_routing(num_tokens, num_logical_experts, num_topk, seed=300 + rank)
        topk_idx_physical = phase1_remapper.remap(topk_idx_logical)

        log(rank, phase1_role,
            f"---- Phase 1 (4 ranks, 2a+2f): active_ranks={phase1_active_ranks.tolist()} ----")
        if phase1_role == "attention":
            print_routing_stats(rank, phase1_role, topk_idx_physical, 4,
                                phase1_num_experts_per_rank, "phase1 routing")

        combined_x, _, _ = run_dispatch_combine(
            m2n, phase1_active_ranks, x, topk_idx_physical, topk_weights, num_topk,
            use_fp8=False,
        )
        if phase1_role == "attention":
            verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x, "Test 3b Phase 1", rank, phase1_role)
        log(rank, phase1_role, "Phase 1 (4 rank, 2a+2f) complete")
    else:
        phase1_group = dist.new_group(phase1_ranks)
        log(rank, "ffn", "Phase 1: waiting (not yet participating)")

    nccl_barrier(group)

    # --- Phase 2: 8 ranks (2a+6f) ---
    phase2_attention = [0, 1]
    phase2_ffn = [2, 3, 4, 5, 6, 7]
    phase2_active_ranks = make_active_ranks(8)

    # Inference engine side: build mapping with replication
    ffn_assignment = {
        2: list(range(0, 64)),
        3: list(range(64, 128)),
        4: list(range(0, 64)),      # replica
        5: list(range(64, 128)),    # replica
        6: list(range(0, 64)),      # replica
        7: list(range(64, 128)),    # replica
    }
    phase2_phy2log = build_phy2log(
        8, phase2_attention, phase2_ffn, num_logical_experts,
        ffn_expert_assignment=ffn_assignment,
    )
    phase2_num_experts_per_rank = len(phase2_phy2log) // 8
    phase2_remapper = ExpertRemapper(phase2_phy2log, num_logical_experts)

    # Mooncake side: use scale_up() for ranks 0-3 (existing M2NBuffer),
    # construct new M2NBuffer for ranks 4-7 (new participants)
    if rank < 4:
        log(rank, m2n.role, "Calling scale_up() to rebuild M2NBuffer for 8-rank topology")
        m2n.scale_up(group, phase2_attention, phase2_ffn, phase2_num_experts_per_rank)
        m2n_p2 = m2n
    else:
        log(rank, "ffn", "Creating new M2NBuffer for 8-rank topology")
        m2n_p2 = M2NBuffer(
            group, phase2_attention, phase2_ffn,
            phase2_num_experts_per_rank, num_max_dispatch_tokens_per_rank, hidden,
        )
    phase2_role = m2n_p2.role

    log(rank, phase2_role,
        f"---- Phase 2 (8 ranks, 2a+6f): num_experts_total={len(phase2_phy2log)}, "
        f"active_ranks={phase2_active_ranks.tolist()} ----")

    x = make_tokens(num_tokens, hidden, rank) if phase2_role == "attention" else torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx_logical, topk_weights = make_routing(num_tokens, num_logical_experts, num_topk, seed=400 + rank)
    topk_idx_physical = phase2_remapper.remap(topk_idx_logical)

    if phase2_role == "attention":
        print_routing_stats(rank, phase2_role, topk_idx_physical, 8,
                            phase2_num_experts_per_rank, "phase2 routing (with replicas)")

    combined_x2, recv_x, packed_recv_count = run_dispatch_combine(
        m2n_p2, phase2_active_ranks, x, topk_idx_physical, topk_weights, num_topk,
        use_fp8=False,
    )

    if phase2_role == "ffn":
        total_recv = packed_recv_count.sum().item()
        is_replica = rank >= 4
        replica_tag = " (replica)" if is_replica else " (original)"
        log(rank, phase2_role,
            f"Phase 2 recv: {int(total_recv)} tokens{replica_tag}, "
            f"counts={packed_recv_count.tolist()}")

    if phase2_role == "attention":
        verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_x2, "Test 3b Phase 2", rank, phase2_role)

    nccl_barrier(group)
    if rank == 0:
        print("=" * 60)
        print("Test 3b: Scaling Up -- Expansion via scale_up() (4->8) -- PASSED")
        print("=" * 60, flush=True)


# ===========================================================================
# Test 4: CUDA Graph Steady-State Capture/Replay
# ===========================================================================

def test_cuda_graph_steady_state(rank: int, num_ranks: int, group: dist.ProcessGroup):
    """
    Test CUDA graph capture and replay with stable rank topology.

    With a stable rank set (no topology changes), capture a full
    dispatch+combine cycle into a CUDA graph, then replay it multiple
    times. The output after each replay must match the eagerly-computed
    reference.

    This verifies that:
    - All Buffer internal state is graph-safe (no host-side branching on
      mutable state between capture and replay)
    - active_ranks tensor pointer is stable (engine updates values, not
      the tensor itself)
    - topk_idx / topk_weights can be updated between replays by writing
      into the same tensor
    """
    num_logical_experts = 128
    hidden = 2560
    num_topk = 8
    num_tokens = 64  # smaller for graph capture
    num_max_dispatch_tokens_per_rank = 64

    attention_ranks = [0, 1]
    ffn_ranks = [2, 3]

    phy2log = build_phy2log(num_ranks, attention_ranks, ffn_ranks, num_logical_experts)
    num_experts_per_rank = len(phy2log) // num_ranks
    remapper = ExpertRemapper(phy2log, num_logical_experts)
    active_ranks = make_active_ranks(num_ranks)

    m2n = M2NBuffer(
        group, attention_ranks, ffn_ranks, num_experts_per_rank,
        num_max_dispatch_tokens_per_rank, hidden,
    )
    role = m2n.role

    # Pre-allocate stable tensors for graph capture
    x = make_tokens(num_tokens, hidden, rank) if role == "attention" else torch.empty(0, hidden, dtype=torch.bfloat16, device="cuda")
    topk_idx_logical, topk_weights = make_routing(num_tokens, num_logical_experts, num_topk, seed=500 + rank)
    topk_idx_physical = remapper.remap(topk_idx_logical)

    if role == "attention":
        log(rank, role,
            f"input x: shape={tuple(x.shape)}, active_ranks={active_ranks.tolist()}")
        print_routing_stats(rank, role, topk_idx_physical, num_ranks, num_experts_per_rank, "warmup routing")

    # --- Warmup: run eagerly first to establish internal buffer state ---
    log(rank, role, "---- Warmup: 3 eager runs ----")
    for _ in range(3):
        combined_ref, _, _ = run_dispatch_combine(
            m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk, use_fp8=False,
        )
    torch.cuda.synchronize()

    if role == "attention":
        verify_combine_correctness(x, topk_idx_physical, topk_weights, combined_ref, "Test 4 (warmup)", rank, role)

    nccl_barrier(group)
    log(rank, role, "Warmup complete, capturing CUDA graph...")

    # --- Capture CUDA graph ---
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    # Placeholder output -- will be filled during capture
    combined_out = torch.zeros_like(combined_ref) if role == "attention" else None

    with torch.cuda.graph(graph, stream=stream):
        if m2n.role == "attention":
            recv_x, packed_recv_count, handle, event, hook = m2n.a2e_isend(
                x, topk_idx_physical, active_ranks, use_fp8=False,
            )
        else:
            recv_x, packed_recv_count, handle, event, hook = m2n.a2e_irecv(
                active_ranks, num_topk, use_fp8=False,
            )

        if m2n.role == "ffn":
            expert_out = recv_x.clone()
            combined_graph, event2, hook2 = m2n.e2a_isend(
                expert_out, active_ranks, handle,
            )
        else:
            combined_graph, event2, hook2 = m2n.e2a_irecv(
                topk_idx_physical, topk_weights, active_ranks, handle,
            )

    log(rank, role, "CUDA graph captured successfully")

    # --- Replay and verify ---
    num_replays = 5
    log(rank, role, f"---- Replay {num_replays} times with same routing ----")
    for i in range(num_replays):
        graph.replay()
        torch.cuda.synchronize()

        if role == "attention":
            diff = calc_diff(combined_ref, combined_graph)
            has_nan = torch.isnan(combined_graph).sum().item()
            log(rank, role, f"Test 4 replay {i}: diff = {diff:.2e}, nan_count = {has_nan}")
            assert has_nan == 0, f"Test 4 replay {i}: NaN in graph output"
            assert diff < 1e-3, f"Test 4 replay {i}: diff {diff} too large"

    # --- Replay with updated routing (same tensor, new values) ---
    log(rank, role, "---- Updating routing in-place via copy_() for graph replay ----")
    topk_idx_logical2, topk_weights2 = make_routing(num_tokens, num_logical_experts, num_topk, seed=600 + rank)
    topk_idx_physical2 = remapper.remap(topk_idx_logical2)
    # Update in-place: graph captures the tensor pointer, not the data
    topk_idx_physical.copy_(topk_idx_physical2)
    topk_weights.copy_(topk_weights2)

    if role == "attention":
        print_routing_stats(rank, role, topk_idx_physical, num_ranks, num_experts_per_rank, "updated routing")

    graph.replay()
    torch.cuda.synchronize()
    log(rank, role, "Graph replayed with updated routing")

    # All ranks must participate in reference dispatch+combine
    combined_ref2, _, _ = run_dispatch_combine(
        m2n, active_ranks, x, topk_idx_physical, topk_weights, num_topk, use_fp8=False,
    )

    if role == "attention":
        diff2 = calc_diff(combined_ref2, combined_graph)
        has_nan2 = torch.isnan(combined_graph).sum().item()
        log(rank, role, f"Test 4 (updated routing): diff = {diff2:.2e}, nan_count = {has_nan2}")
        assert has_nan2 == 0, "Test 4 updated routing: NaN"
        assert diff2 < 1e-3, f"Test 4 updated routing: diff {diff2} too large"

    nccl_barrier(group)
    if rank == 0:
        print("=" * 60)
        print("Test 4: CUDA Graph Steady-State -- PASSED")
        print("=" * 60, flush=True)


# ===========================================================================
# Main entry points
# ===========================================================================

def test_loop_4rank(local_rank: int, num_local_ranks: int):
    """Run Test 1, 2, 3a, 4 with 4 ranks."""
    rank, num_ranks, group, cpu_group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == 4, f"Expected 4 ranks, got {num_ranks}"

    print(f"[Rank {rank}] Starting 4-rank tests...", flush=True)

    test_basic_m2n(rank, num_ranks, group)
    test_scaling_down(rank, num_ranks, group)
    test_scaling_up_recovery(rank, num_ranks, group)
    test_cuda_graph_steady_state(rank, num_ranks, group)

    try:
        dist.destroy_process_group()
    except Exception:
        pass


def test_loop_8rank(local_rank: int, num_local_ranks: int):
    """Run Test 3b with 8 ranks."""
    rank, num_ranks, group, cpu_group = init_dist(local_rank, num_local_ranks)
    assert num_ranks == 8, f"Expected 8 ranks, got {num_ranks}"

    print(f"[Rank {rank}] Starting 8-rank expansion test...", flush=True)

    test_scaling_up_expansion(rank, num_ranks, group)

    try:
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    num_processes = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    if num_processes == 4:
        print("Running Test 1, 2, 3a, 4 with 4 ranks...")
        torch.multiprocessing.spawn(test_loop_4rank, args=(num_processes,), nprocs=num_processes)
    elif num_processes == 8:
        print("Running Test 3b with 8 ranks...")
        torch.multiprocessing.spawn(test_loop_8rank, args=(num_processes,), nprocs=num_processes)
    else:
        print(f"Usage: python {sys.argv[0]} [4|8]")
        print("  4 -- Run Test 1, 2, 3a, 4 (basic, scaling down, recovery, CUDA graph)")
        print("  8 -- Run Test 3b (expansion 4->8 rank via scale_up)")
        sys.exit(1)
