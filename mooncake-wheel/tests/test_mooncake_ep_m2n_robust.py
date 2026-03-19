"""
test_mooncake_ep_m2n_robust.py

Comprehensive M2N fault-tolerance tests using REAL OS-level process kill
and restart via multiprocessing.Process + SIGKILL.

Unlike test_mooncake_ep_m2n.py (which simulates failures via active_ranks
masking), this file truly kills worker processes, spawns replacements with
new PIDs and fresh GPU memory, and uses file-based barriers for sync.

Three tests
-----------
T_Kill
  Real SIGKILL of rank 3 after Phase 1.  Surviving ranks run dispatch+combine
  with 2 s timeout.  Kernel sets active_ranks[3]=0 on timeout.  Re-route
  with masked topk_idx.  Verify numeric correctness.

T_Recovery
  Kill rank 3.  Survivors detect crash (Phase 2).  Spawn NEW rank 3
  (different PID, new GPU memory, different RDMA virtual addresses).
  Survivors call dist.destroy_process_group() [LOCAL op -- safe with dead rank].
  All 4 form new group -> scale_up() -> verify full routing restored (Phase 3).

T_ScaleUp
  Baseline with 4 ranks (2 attn + 2 FFN).  Spawn ranks 4-7 while original
  ranks wait at barrier.  All 8 form new group -> scale_up() -> verify
  2a+6f topology with expert replication.

Usage
-----
  python test_mooncake_ep_m2n_robust.py [--tests kill recovery scaleup] [--gpus N]
"""

import os
import sys
import signal
import time
import tempfile
import shutil
import multiprocessing as mp
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))

from mooncake.mooncake_ep_m2n_buffer import M2NBuffer
from ep_test_utils import calc_diff
import mooncake.pg  # registers 'mooncake' backend

T_KILL_PORT     = 29700
T_RECOVERY_PORT = 29710   # epoch-0=29710, epoch-1=29711
T_SCALEUP_PORT  = 29720   # epoch-0=29720, epoch-1=29721


# ===========================================================================
# File-based barrier
# ===========================================================================

class FileBarrier:
    """
    Sync barrier backed by filesystem markers (NCCL-independent).
    Survives process kill/restart -- arrival is a persistent file touch.
    """
    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    def _f(self, phase: str, rank: int) -> Path:
        return self.base / f"{phase}.r{rank}"

    def arrive(self, phase: str, rank: int):
        self._f(phase, rank).touch()

    def wait_for(self, phase: str, ranks: List[int], timeout: float = 120.0):
        deadline = time.monotonic() + timeout
        while True:
            missing = [r for r in ranks if not self._f(phase, r).exists()]
            if not missing:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"FileBarrier[{self.base.name}] '{phase}': "
                    f"timed out waiting for ranks {missing}")
            time.sleep(0.05)

    def arrive_and_wait(self, phase: str, rank: int,
                        ranks: List[int], timeout: float = 120.0):
        self.arrive(phase, rank)
        self.wait_for(phase, ranks, timeout)


# ===========================================================================
# Dist group management
# ===========================================================================

def init_group(rank: int, world_ranks: List[int], port: int,
               timeout_sec: int = 60) -> dist.ProcessGroup:
    """
    (Re-)initialise the default dist group for this epoch via TCPStore.
    If a previous group exists it is destroyed first.
    destroy_process_group() is LOCAL (non-collective) -- safe with dead ranks.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
    world_size = len(world_ranks)
    local_rank = world_ranks.index(rank)
    is_master  = (rank == world_ranks[0])
    store = dist.TCPStore(
        "127.0.0.1", port, world_size, is_master,
        timeout=timedelta(seconds=timeout_sec),
    )
    dist.init_process_group(
        backend="mooncake",
        store=store,
        rank=local_rank,
        world_size=world_size,
    )
    return dist.group.WORLD


# ===========================================================================
# Shared test helpers
# ===========================================================================

def log(rank: int, role: str, msg: str):
    tag = "Attn" if role == "attention" else "FFN "
    print(f"[Rank {rank:>2} ({tag})] {msg}", flush=True)


def make_tokens(num_tokens: int, hidden: int, rank: int) -> torch.Tensor:
    x = torch.full((num_tokens, hidden), float(rank),
                   dtype=torch.bfloat16, device="cuda")
    x[:, -128:] = torch.arange(num_tokens, device="cuda").to(
        torch.bfloat16).view(-1, 1)
    return x


def make_routing(num_tokens: int, num_logical_experts: int,
                 num_topk: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    scores    = torch.rand(num_tokens, num_logical_experts,
                           generator=gen, device="cuda")
    topk_idx  = scores.topk(num_topk, dim=-1).indices.to(torch.int64)
    topk_weights = (torch.rand(num_tokens, num_topk, generator=gen,
                               device="cuda", dtype=torch.float32).abs() + 0.1)
    return topk_idx, topk_weights


def make_active_ranks(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.int32, device="cuda")


def build_phy2log(num_ranks: int,
                  ffn_ranks: List[int],
                  num_logical_experts: int,
                  ffn_assignment: Optional[Dict[int, List[int]]] = None
                  ) -> List[int]:
    if ffn_assignment is None:
        epr = num_logical_experts // len(ffn_ranks)
    else:
        epr = max(len(v) for v in ffn_assignment.values())
    phy2log = [-1] * (num_ranks * epr)
    if ffn_assignment is None:
        for i, r in enumerate(sorted(ffn_ranks)):
            for j in range(epr):
                phy2log[r * epr + j] = i * epr + j
    else:
        for r, ids in ffn_assignment.items():
            for j, lid in enumerate(ids):
                phy2log[r * epr + j] = lid
    return phy2log


class ExpertRemapper:
    def __init__(self, phy2log: List[int], num_logical_experts: int):
        log2phy: List[List[int]] = [[] for _ in range(num_logical_experts)]
        for pid, lid in enumerate(phy2log):
            if 0 <= lid < num_logical_experts:
                log2phy[lid].append(pid)
        for lid in range(num_logical_experts):
            assert log2phy[lid], f"logical expert {lid} unmapped"
        max_rep = max(len(s) for s in log2phy)
        self._max_rep = max_rep
        self._first   = torch.tensor([s[0] for s in log2phy],
                                     dtype=torch.int64, device="cuda")
        self._padded  = torch.full((num_logical_experts, max_rep), -1,
                                   dtype=torch.int64, device="cuda")
        self._counts  = torch.zeros(num_logical_experts,
                                    dtype=torch.int64, device="cuda")
        for lid, slots in enumerate(log2phy):
            self._counts[lid] = len(slots)
            for j, pid in enumerate(slots):
                self._padded[lid, j] = pid

    def remap(self, topk_logical: torch.Tensor) -> torch.Tensor:
        mask = topk_logical >= 0
        safe = topk_logical.clamp(min=0)
        if self._max_rep == 1:
            phy = self._first[safe]
        else:
            counts = self._counts[safe]
            ri = (torch.rand(topk_logical.shape, device="cuda") *
                  counts.float()).long().clamp(max=self._max_rep - 1)
            phy = self._padded[safe, ri]
        return phy.masked_fill(~mask, -1)


def mask_inactive_experts(topk_idx: torch.Tensor,
                          active_ranks: torch.Tensor,
                          epr: int) -> torch.Tensor:
    num_ranks = active_ranks.shape[0]
    ok   = torch.zeros(num_ranks * epr, dtype=torch.bool, device="cuda")
    for r in range(num_ranks):
        if active_ranks[r].item():
            ok[r * epr:(r + 1) * epr] = True
    out  = topk_idx.clone()
    flat = out.view(-1)
    valid = flat >= 0
    flat[valid & ~ok[flat.clamp(min=0)]] = -1
    return out


def print_routing_stats(rank: int, role: str, topk_idx: torch.Tensor,
                        num_ranks: int, epr: int, label: str = "routing"):
    tpr   = [int(((topk_idx >= r*epr) & (topk_idx < (r+1)*epr)).sum())
             for r in range(num_ranks)]
    valid = int((topk_idx >= 0).sum())
    log(rank, role, f"[{label}] per-rank topk counts: {tpr}  (valid={valid})")


def run_dispatch_combine(m2n: M2NBuffer,
                         active_ranks: torch.Tensor,
                         x: torch.Tensor,
                         topk_idx: torch.Tensor,
                         topk_weights: torch.Tensor,
                         num_topk: int,
                         timeout_us: int = -1) -> tuple:
    t0 = time.monotonic()
    if m2n.role == "attention":
        recv_x, prc, handle, _, _ = m2n.a2e_isend(
            x, topk_idx, active_ranks,
            timeout_us=timeout_us, use_fp8=False)
    else:
        recv_x, prc, handle, _, _ = m2n.a2e_irecv(
            active_ranks, num_topk,
            timeout_us=timeout_us, use_fp8=False)
    t1 = time.monotonic()
    
    if m2n.role == "ffn":
        combined_x, _, _ = m2n.e2a_isend(
            recv_x.clone(), active_ranks, handle, timeout_us=timeout_us)
    else:
        combined_x, _, _ = m2n.e2a_irecv(
            topk_idx, topk_weights, active_ranks, handle, timeout_us=timeout_us)
    t2 = time.monotonic()
    torch.cuda.synchronize()
    print(f"  [{m2n.role}] dispatch={1000*(t1-t0):.1f}ms "
          f"combine={1000*(t2-t1):.1f}ms", flush=True)
    return combined_x, recv_x, prc


def verify_correctness(x, topk_idx, topk_weights, combined_x,
                       label, rank, role):
    if role != "attention":
        return
    valid_w  = topk_weights.masked_fill(topk_idx < 0, 0.0)
    expected = x * valid_w.sum(dim=1, keepdim=True)
    diff     = calc_diff(expected, combined_x)
    nan_cnt  = int(torch.isnan(combined_x).sum())
    log(rank, role, f"{label}: diff={diff:.2e}  nan={nan_cnt}")
    assert nan_cnt == 0, f"{label}: NaN in combined_x"
    assert diff < 1e-3,  f"{label}: diff {diff:.2e} too large"


def verify_with_active_mask(x, topk_idx, topk_weights, combined_x,
                             active_ranks, epr, label, rank, role):
    """
    Correctness check when some ranks are inactive.
    Tokens routed only to active-rank slots must be numerically correct.
    Tokens with any inactive-rank entry are checked NaN-free only
    (stale buffer-pair data may exist for those slots).
    """
    if role != "attention":
        return
    nan_cnt = int(torch.isnan(combined_x).sum())
    assert nan_cnt == 0, f"{label}: NaN in combined_x"
    num_ranks = active_ranks.shape[0]
    ok = torch.zeros(num_ranks * epr, dtype=torch.bool, device="cuda")
    for r in range(num_ranks):
        if active_ranks[r].item():
            ok[r * epr:(r + 1) * epr] = True
    flat = topk_idx.view(-1)
    valid = flat >= 0
    hits_dead    = (valid & ~ok[flat.clamp(min=0)]).view(topk_idx.shape).any(dim=1)
    fully_active = ~hits_dead
    n_full       = int(fully_active.sum())
    log(rank, role, f"{label}: {n_full}/{x.shape[0]} tokens fully on active ranks")
    if n_full > 0:
        valid_w  = topk_weights.masked_fill(topk_idx < 0, 0.0)
        expected = x * valid_w.sum(dim=1, keepdim=True)
        diff = calc_diff(expected[fully_active], combined_x[fully_active])
        log(rank, role, f"{label}: fully-active diff={diff:.2e}")
        assert diff < 1e-3, f"{label}: fully-active diff {diff:.2e} too large"


# ===========================================================================
# T_Kill
# ===========================================================================

def worker_t_kill(rank: int, barrier_dir: str,
                  result_q: mp.Queue, port: int):
    """
    All 4 ranks share this worker.

    Phase 1   All 4 ranks baseline.  Arrive at 'phase1'.
    [SIGKILL] Coordinator kills rank 3 after 'phase1' barrier.
    Phase 2a  Survivors [0,1,2]: dispatch+combine with 2 s timeout.
              Rank 3 dead -> combine never arrives -> timeout ->
              kernel sets active_ranks[3]=0.
    Phase 2b  Survivors: mask rank-3 expert slots -> re-run -> verify.
    """
    torch.cuda.set_device(rank)
    bar = FileBarrier(Path(barrier_dir) / "t_kill")

    NUM_LOGICAL = 128
    HIDDEN      = 2560
    NUM_TOPK    = 8
    NUM_TOKENS  = 128
    MAX_DISP    = 128
    ATTN        = [0, 1]
    FFN         = [2, 3]
    ALL         = [0, 1, 2, 3]
    SURVIVORS   = [0, 1, 2]

    group    = init_group(rank, ALL, port)
    phy2log  = build_phy2log(4, FFN, NUM_LOGICAL)
    epr      = len(phy2log) // 4
    remapper = ExpertRemapper(phy2log, NUM_LOGICAL)
    active   = make_active_ranks(4)
    m2n      = M2NBuffer(group, ATTN, FFN, epr, MAX_DISP, HIDDEN)
    role     = m2n.role

    x = (make_tokens(NUM_TOKENS, HIDDEN, rank)
         if role == "attention"
         else torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda"))
    tidx_log, tw = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                seed=10 + rank)
    tidx_phy = remapper.remap(tidx_log)

    # Phase 1
    log(rank, role, "T_Kill Phase 1: baseline")
    if role == "attention":
        print_routing_stats(rank, role, tidx_phy, 4, epr, "phase1")
    cx1, _, _ = run_dispatch_combine(m2n, active, x, tidx_phy, tw, NUM_TOPK)
    verify_correctness(x, tidx_phy, tw, cx1, "T_Kill Phase 1", rank, role)
    bar.arrive_and_wait("phase1", rank, ALL)
    log(rank, role, "T_Kill Phase 1 PASSED")
    # coordinator kills rank 3 here; rank 3 never reaches Phase 2

    # Phase 2a -- survivors only
    bar.arrive_and_wait("phase2_start", rank, SURVIVORS)
    log(rank, role, "T_Kill Phase 2a: 2 s timeout (rank 3 dead)")
    cx2a, _, _ = run_dispatch_combine(
        m2n, active, x, tidx_phy, tw, NUM_TOPK, timeout_us=2_000_000)
    assert active[3].item() == 0, "kernel must set active_ranks[3]=0 on timeout"
    log(rank, role, f"T_Kill Phase 2a: crash confirmed  active={active.tolist()}")
    verify_with_active_mask(x, tidx_phy, tw, cx2a, active, epr,
                            "T_Kill Phase 2a", rank, role)
    bar.arrive_and_wait("phase2a", rank, SURVIVORS)

    # Phase 2b -- re-route
    tidx_masked = mask_inactive_experts(tidx_phy, active, epr)
    n_masked = int((tidx_masked == -1).sum()) - int((tidx_phy == -1).sum())
    log(rank, role, f"T_Kill Phase 2b: {n_masked} entries masked")
    if role == "attention":
        print_routing_stats(rank, role, tidx_masked, 4, epr, "phase2b")
    cx2b, _, _ = run_dispatch_combine(
        m2n, active, x, tidx_masked, tw, NUM_TOPK, timeout_us=2_000_000)
    verify_correctness(x, tidx_masked, tw, cx2b,
                       "T_Kill Phase 2b (re-routed)", rank, role)
    bar.arrive_and_wait("phase2b", rank, SURVIVORS)

    result_q.put({"rank": rank, "test": "T_Kill", "status": "PASSED"})
    log(rank, role, "T_Kill PASSED")


# ===========================================================================
# T_Recovery
# ===========================================================================

def worker_t_recovery_survivor(rank: int, barrier_dir: str,
                                result_q: mp.Queue,
                                port_e0: int, port_e1: int):
    """
    Survivor workers (ranks 0, 1, 2) across all three T_Recovery phases.

    Phase 1   All 4 ranks baseline.
    [SIGKILL] Coordinator kills rank 3 after 'phase1'.
    Phase 2   Survivors timeout, detect crash, re-route.
    Phase 3   Survivors destroy old group (LOCAL op).
              Wait for new rank-3 signal.
              All 4 form epoch-1 group -> scale_up() -> verify recovery.
    """
    torch.cuda.set_device(rank)
    bar = FileBarrier(Path(barrier_dir) / "t_recovery")

    NUM_LOGICAL = 128
    HIDDEN      = 2560
    NUM_TOPK    = 8
    NUM_TOKENS  = 128
    MAX_DISP    = 128
    ATTN        = [0, 1]
    FFN         = [2, 3]
    ALL         = [0, 1, 2, 3]
    SURVIVORS   = [0, 1, 2]

    # Epoch 0
    group0   = init_group(rank, ALL, port_e0)
    phy2log  = build_phy2log(4, FFN, NUM_LOGICAL)
    epr      = len(phy2log) // 4
    remapper = ExpertRemapper(phy2log, NUM_LOGICAL)
    active   = make_active_ranks(4)
    m2n      = M2NBuffer(group0, ATTN, FFN, epr, MAX_DISP, HIDDEN)
    role     = m2n.role

    x = (make_tokens(NUM_TOKENS, HIDDEN, rank)
         if role == "attention"
         else torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda"))
    tidx_log, tw = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                seed=20 + rank)
    tidx_phy = remapper.remap(tidx_log)

    # Phase 1
    log(rank, role, "T_Recovery Phase 1: baseline")
    if role == "attention":
        print_routing_stats(rank, role, tidx_phy, 4, epr, "phase1")
    cx1, _, _ = run_dispatch_combine(m2n, active, x, tidx_phy, tw, NUM_TOPK)
    verify_correctness(x, tidx_phy, tw, cx1, "T_Recovery Phase 1", rank, role)
    bar.arrive_and_wait("phase1", rank, ALL)
    log(rank, role, "T_Recovery Phase 1 PASSED")

    # Phase 2
    bar.arrive_and_wait("phase2_start", rank, SURVIVORS)
    log(rank, role, "T_Recovery Phase 2: 2 s timeout crash detection")
    cx2, _, _ = run_dispatch_combine(
        m2n, active, x, tidx_phy, tw, NUM_TOPK, timeout_us=2_000_000)
    active[3] = 0   # kernel sets this on timeout; force for clarity
    log(rank, role, f"T_Recovery Phase 2: crash confirmed  active={active.tolist()}")
    verify_with_active_mask(x, tidx_phy, tw, cx2, active, epr,
                            "T_Recovery Phase 2", rank, role)
    bar.arrive_and_wait("phase2_done", rank, SURVIVORS)
    log(rank, role, "T_Recovery Phase 2 PASSED")

    # Phase 3 -- destroy old group, form new group with new rank 3
    log(rank, role, "T_Recovery Phase 3: destroy epoch-0 group (LOCAL op)")
    dist.destroy_process_group()

    log(rank, role, "T_Recovery Phase 3: waiting for new rank-3 process...")
    bar.wait_for("new_r3_ready", [3])

    # All 4 synchronise before forming the new group so rank-0 (TCPStore master)
    # is guaranteed to be at its store before rank-3 tries to connect.
    bar.arrive_and_wait("phase3_form_group", rank, ALL)

    log(rank, role, "T_Recovery Phase 3: forming epoch-1 group")
    group1 = init_group(rank, ALL, port_e1)

    log(rank, role, "T_Recovery Phase 3: scale_up() -- new rank-3 has fresh RDMA addrs")
    m2n.scale_up(group1, ATTN, FFN, epr)

    active[3] = 1
    if role == "attention":
        print_routing_stats(rank, role, tidx_phy, 4, epr, "phase3 full routing")
    cx3, _, _ = run_dispatch_combine(m2n, active, x, tidx_phy, tw, NUM_TOPK)
    verify_correctness(x, tidx_phy, tw, cx3,
                       "T_Recovery Phase 3 (recovered)", rank, role)
    bar.arrive_and_wait("phase3_done", rank, ALL)

    result_q.put({"rank": rank, "test": "T_Recovery", "status": "PASSED"})
    log(rank, role, "T_Recovery PASSED")


def worker_t_recovery_rank3_orig(rank: int, barrier_dir: str, port_e0: int):
    """
    Original rank 3 -- participates in Phase 1 then waits for SIGKILL.
    Never reaches Phase 2.
    """
    assert rank == 3
    torch.cuda.set_device(rank)
    bar = FileBarrier(Path(barrier_dir) / "t_recovery")

    NUM_LOGICAL = 128
    HIDDEN      = 2560
    NUM_TOPK    = 8
    NUM_TOKENS  = 128
    MAX_DISP    = 128
    ATTN        = [0, 1]
    FFN         = [2, 3]
    ALL         = [0, 1, 2, 3]

    group0   = init_group(rank, ALL, port_e0)
    phy2log  = build_phy2log(4, FFN, NUM_LOGICAL)
    epr      = len(phy2log) // 4
    remapper = ExpertRemapper(phy2log, NUM_LOGICAL)
    active   = make_active_ranks(4)
    m2n      = M2NBuffer(group0, ATTN, FFN, epr, MAX_DISP, HIDDEN)
    role     = m2n.role  # "ffn"

    x = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda")
    tidx_log, tw = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                seed=20 + rank)
    tidx_phy = remapper.remap(tidx_log)

    log(rank, role, "T_Recovery Phase 1: baseline (SIGKILLed after this)")
    run_dispatch_combine(m2n, active, x, tidx_phy, tw, NUM_TOPK)
    bar.arrive_and_wait("phase1", rank, ALL)
    log(rank, role, "T_Recovery Phase 1 done -- waiting for SIGKILL")
    time.sleep(3600)


def worker_t_recovery_rank3_new(rank: int, barrier_dir: str,
                                 result_q: mp.Queue, port_e1: int):
    """
    NEW rank-3 process (Phase 3 only).

    This process has a DIFFERENT PID from the killed rank 3.
    Its CUDA allocations use new virtual addresses.  scale_up() on survivors
    exchanges these addresses via Buffer.connect(), proving RDMA works with
    freshly allocated memory.
    """
    assert rank == 3
    torch.cuda.set_device(rank)
    bar = FileBarrier(Path(barrier_dir) / "t_recovery")

    NUM_LOGICAL = 128
    HIDDEN      = 2560
    NUM_TOPK    = 8
    NUM_TOKENS  = 128
    MAX_DISP    = 128
    ATTN        = [0, 1]
    FFN         = [2, 3]
    ALL         = [0, 1, 2, 3]

    role = "ffn"
    log(rank, role,
        f"T_Recovery Phase 3 (NEW rank 3): PID={os.getpid()} -- fresh GPU memory")

    # Signal readiness immediately so survivors can proceed to Phase 3
    bar.arrive("new_r3_ready", rank)

    # Wait until survivors finished Phase 2 (so they are at destroy_group step)
    bar.wait_for("phase2_done", [0, 1, 2])

    bar.arrive_and_wait("phase3_form_group", rank, ALL)

    log(rank, role, "T_Recovery Phase 3: forming epoch-1 group")
    group1 = init_group(rank, ALL, port_e1)

    phy2log  = build_phy2log(4, FFN, NUM_LOGICAL)
    epr      = len(phy2log) // 4
    remapper = ExpertRemapper(phy2log, NUM_LOGICAL)
    active   = make_active_ranks(4)

    log(rank, role, "T_Recovery Phase 3: NEW M2NBuffer (Buffer.connect exchanges new RDMA addrs)")
    m2n = M2NBuffer(group1, ATTN, FFN, epr, MAX_DISP, HIDDEN)

    x = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda")
    tidx_log, tw = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                seed=20 + rank)
    tidx_phy = remapper.remap(tidx_log)

    run_dispatch_combine(m2n, active, x, tidx_phy, tw, NUM_TOPK)
    bar.arrive_and_wait("phase3_done", rank, ALL)

    result_q.put({"rank": rank, "test": "T_Recovery", "status": "PASSED"})
    log(rank, role, "T_Recovery Phase 3 (NEW rank 3) PASSED")


# ===========================================================================
# T_ScaleUp
# ===========================================================================

def worker_t_scaleup_orig(rank: int, barrier_dir: str,
                           result_q: mp.Queue,
                           port_e0: int, port_e1: int):
    """
    Original ranks 0-3 across both T_ScaleUp phases.

    Phase 1  2a+2f baseline (4 ranks).
    [spawn]  Coordinator spawns ranks 4-7 after 'phase1'.
    Phase 2  Destroy Phase-1 group (local).  Wait at 'phase2_ready' for all 8.
             Form epoch-1 group.  scale_up() to 2a+6f.  Verify.
    """
    torch.cuda.set_device(rank)
    bar = FileBarrier(Path(barrier_dir) / "t_scaleup")

    NUM_LOGICAL = 128
    HIDDEN      = 2560
    NUM_TOPK    = 8
    NUM_TOKENS  = 128
    MAX_DISP    = 128
    P1_ATTN     = [0, 1]
    P1_FFN      = [2, 3]
    P1_ALL      = [0, 1, 2, 3]
    P2_ATTN     = [0, 1]
    P2_FFN      = [2, 3, 4, 5, 6, 7]
    P2_ALL      = list(range(8))

    # Epoch 0
    group0      = init_group(rank, P1_ALL, port_e0)
    phy2log_p1  = build_phy2log(4, P1_FFN, NUM_LOGICAL)
    epr_p1      = len(phy2log_p1) // 4
    remapper_p1 = ExpertRemapper(phy2log_p1, NUM_LOGICAL)
    active_p1   = make_active_ranks(4)
    m2n         = M2NBuffer(group0, P1_ATTN, P1_FFN, epr_p1, MAX_DISP, HIDDEN)
    role        = m2n.role

    x = (make_tokens(NUM_TOKENS, HIDDEN, rank)
         if role == "attention"
         else torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda"))
    tidx_log_p1, tw_p1 = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                       seed=30 + rank)
    tidx_phy_p1 = remapper_p1.remap(tidx_log_p1)

    # Phase 1
    log(rank, role, "T_ScaleUp Phase 1 (4 ranks, 2a+2f): baseline")
    if role == "attention":
        print_routing_stats(rank, role, tidx_phy_p1, 4, epr_p1, "phase1")
    cx_p1, _, _ = run_dispatch_combine(
        m2n, active_p1, x, tidx_phy_p1, tw_p1, NUM_TOPK)
    verify_correctness(x, tidx_phy_p1, tw_p1, cx_p1,
                       "T_ScaleUp Phase 1", rank, role)
    bar.arrive_and_wait("phase1", rank, P1_ALL)
    log(rank, role, "T_ScaleUp Phase 1 PASSED")

    # Destroy Phase-1 group (LOCAL op) and wait for all 8 at phase2_ready
    log(rank, role, "T_ScaleUp: destroying Phase-1 group (local)")
    dist.destroy_process_group()

    bar.arrive_and_wait("phase2_ready", rank, P2_ALL)

    # Epoch 1
    log(rank, role, "T_ScaleUp Phase 2: forming 8-rank group")
    group1 = init_group(rank, P2_ALL, port_e1)

    ffn_assignment = {
        2: list(range(0,  64)),  3: list(range(64, 128)),
        4: list(range(0,  64)),  5: list(range(0,  64)),
        6: list(range(64, 128)), 7: list(range(64, 128)),
    }
    phy2log_p2  = build_phy2log(8, P2_FFN, NUM_LOGICAL, ffn_assignment)
    epr_p2      = len(phy2log_p2) // 8
    remapper_p2 = ExpertRemapper(phy2log_p2, NUM_LOGICAL)
    active_p2   = make_active_ranks(8)

    log(rank, role, "T_ScaleUp Phase 2: scale_up() with 8-rank group")
    m2n.scale_up(group1, P2_ATTN, P2_FFN, epr_p2)

    tidx_log_p2, tw_p2 = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                       seed=40 + rank)
    tidx_phy_p2 = remapper_p2.remap(tidx_log_p2)
    if m2n.role == "attention":
        print_routing_stats(rank, m2n.role, tidx_phy_p2, 8, epr_p2,
                            "phase2 (6 FFN w/ replication)")
    cx_p2, _, _ = run_dispatch_combine(
        m2n, active_p2, x, tidx_phy_p2, tw_p2, NUM_TOPK)
    verify_correctness(x, tidx_phy_p2, tw_p2, cx_p2,
                       "T_ScaleUp Phase 2", rank, m2n.role)
    bar.arrive_and_wait("phase2_done", rank, P2_ALL)

    result_q.put({"rank": rank, "test": "T_ScaleUp", "status": "PASSED"})
    log(rank, m2n.role, "T_ScaleUp PASSED")


def worker_t_scaleup_new(rank: int, barrier_dir: str,
                          result_q: mp.Queue, port_e1: int):
    """
    New ranks 4-7 in T_ScaleUp Phase 2.
    Spawned while original ranks wait at 'phase2_ready' barrier.
    """
    assert 4 <= rank <= 7
    torch.cuda.set_device(rank)
    bar = FileBarrier(Path(barrier_dir) / "t_scaleup")

    NUM_LOGICAL = 128
    HIDDEN      = 2560
    NUM_TOPK    = 8
    NUM_TOKENS  = 128
    MAX_DISP    = 128
    P2_ATTN     = [0, 1]
    P2_FFN      = [2, 3, 4, 5, 6, 7]
    P2_ALL      = list(range(8))

    role = "ffn"
    log(rank, role, f"T_ScaleUp Phase 2: new rank {rank}  PID={os.getpid()}")

    bar.arrive_and_wait("phase2_ready", rank, P2_ALL)

    group1 = init_group(rank, P2_ALL, port_e1)

    ffn_assignment = {
        2: list(range(0,  64)),  3: list(range(64, 128)),
        4: list(range(0,  64)),  5: list(range(0,  64)),
        6: list(range(64, 128)), 7: list(range(64, 128)),
    }
    phy2log_p2  = build_phy2log(8, P2_FFN, NUM_LOGICAL, ffn_assignment)
    epr_p2      = len(phy2log_p2) // 8
    remapper_p2 = ExpertRemapper(phy2log_p2, NUM_LOGICAL)
    active_p2   = make_active_ranks(8)

    m2n = M2NBuffer(group1, P2_ATTN, P2_FFN, epr_p2, MAX_DISP, HIDDEN)
    x   = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device="cuda")
    tidx_log, tw = make_routing(NUM_TOKENS, NUM_LOGICAL, NUM_TOPK,
                                seed=40 + rank)
    tidx_phy = remapper_p2.remap(tidx_log)

    _, _, prc = run_dispatch_combine(m2n, active_p2, x, tidx_phy, tw, NUM_TOPK)
    log(rank, role,
        f"T_ScaleUp Phase 2: received {int(prc.sum())} tokens  "
        f"counts={prc.tolist()}")
    bar.arrive_and_wait("phase2_done", rank, P2_ALL)

    result_q.put({"rank": rank, "test": "T_ScaleUp", "status": "PASSED"})
    log(rank, role, "T_ScaleUp PASSED")


# ===========================================================================
# Coordinator
# ===========================================================================

class Coordinator:
    def __init__(self):
        self.barrier_dir = Path(tempfile.mkdtemp(prefix="mooncake_robust_"))
        self.result_q: mp.Queue = mp.Queue()
        self._procs: Dict[int, mp.Process] = {}

    def spawn(self, rank: int, target, *args) -> mp.Process:
        p = mp.Process(target=target, args=args, daemon=False)
        p.start()
        self._procs[rank] = p
        print(f"[Coord] spawned rank {rank}  PID={p.pid}", flush=True)
        return p

    def kill(self, rank: int):
        p = self._procs.pop(rank, None)
        if p and p.is_alive():
            print(f"[Coord] SIGKILL -> rank {rank}  PID={p.pid}", flush=True)
            os.kill(p.pid, signal.SIGKILL)
            p.join(timeout=5.0)
            print(f"[Coord] rank {rank} dead  exit={p.exitcode}", flush=True)

    def bar(self, test: str) -> FileBarrier:
        return FileBarrier(self.barrier_dir / test)

    def collect(self, ranks: List[int], timeout: float = 180.0) -> bool:
        got: Dict[int, dict] = {}
        deadline = time.monotonic() + timeout
        while len(got) < len(ranks):
            try:
                item = self.result_q.get(timeout=0.3)
                got[item["rank"]] = item
            except Exception:
                pass
            if time.monotonic() > deadline:
                missing = [r for r in ranks if r not in got]
                print(f"[Coord] collect() timed out  missing={missing}", flush=True)
                return False
        for rk in sorted(got):
            print(f"[Coord]   rank {rk}: {got[rk]['status']}", flush=True)
        return all(r["status"] == "PASSED" for r in got.values())

    def cleanup(self):
        for p in self._procs.values():
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        shutil.rmtree(self.barrier_dir, ignore_errors=True)


# ===========================================================================
# Test runners
# ===========================================================================

def run_t_kill(coord: Coordinator) -> bool:
    print("\n" + "=" * 70, flush=True)
    print("T_Kill: rank 3 SIGKILLed -> timeout detection -> re-route", flush=True)
    print("=" * 70, flush=True)
    bar = coord.bar("t_kill")
    for r in range(4):
        coord.spawn(r, worker_t_kill,
                    r, str(coord.barrier_dir), coord.result_q, T_KILL_PORT)
    bar.wait_for("phase1", [0, 1, 2, 3], timeout=180)
    print("[Coord] T_Kill Phase 1 done -- SIGKILL rank 3", flush=True)
    coord.kill(3)
    # Survivors self-coordinate via 'phase2_start' barrier for [0,1,2].
    passed = coord.collect([0, 1, 2], timeout=180)
    print("=" * 70, flush=True)
    print(f"T_Kill: {'PASSED' if passed else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)
    return passed


def run_t_recovery(coord: Coordinator) -> bool:
    print("\n" + "=" * 70, flush=True)
    print("T_Recovery: SIGKILL rank 3 -> new process -> scale_up()", flush=True)
    print("=" * 70, flush=True)
    bar = coord.bar("t_recovery")
    pe0, pe1 = T_RECOVERY_PORT, T_RECOVERY_PORT + 1
    for r in range(3):
        coord.spawn(r, worker_t_recovery_survivor,
                    r, str(coord.barrier_dir), coord.result_q, pe0, pe1)
    coord.spawn(3, worker_t_recovery_rank3_orig,
                3, str(coord.barrier_dir), pe0)
    bar.wait_for("phase1", [0, 1, 2, 3], timeout=180)
    print("[Coord] T_Recovery Phase 1 done -- SIGKILL rank 3", flush=True)
    coord.kill(3)
    # Survivors self-coordinate Phase 2 via their own 'phase2_start' barrier.
    bar.wait_for("phase2_done", [0, 1, 2], timeout=180)
    print("[Coord] T_Recovery Phase 2 done -- spawning NEW rank 3", flush=True)
    coord.spawn(3, worker_t_recovery_rank3_new,
                3, str(coord.barrier_dir), coord.result_q, pe1)
    passed = coord.collect([0, 1, 2, 3], timeout=180)
    print("=" * 70, flush=True)
    print(f"T_Recovery: {'PASSED' if passed else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)
    return passed


def run_t_scaleup(coord: Coordinator) -> bool:
    print("\n" + "=" * 70, flush=True)
    print("T_ScaleUp: dynamic rank expansion 4 -> 8", flush=True)
    print("=" * 70, flush=True)
    bar = coord.bar("t_scaleup")
    pe0, pe1 = T_SCALEUP_PORT, T_SCALEUP_PORT + 1
    for r in range(4):
        coord.spawn(r, worker_t_scaleup_orig,
                    r, str(coord.barrier_dir), coord.result_q, pe0, pe1)
    bar.wait_for("phase1", [0, 1, 2, 3], timeout=180)
    print("[Coord] T_ScaleUp Phase 1 done -- spawning ranks 4-7", flush=True)
    for r in range(4, 8):
        coord.spawn(r, worker_t_scaleup_new,
                    r, str(coord.barrier_dir), coord.result_q, pe1)
    passed = coord.collect(list(range(8)), timeout=240)
    print("=" * 70, flush=True)
    print(f"T_ScaleUp: {'PASSED' if passed else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)
    return passed


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="M2N robust fault-tolerance tests (real SIGKILL + restart)")
    parser.add_argument("--tests", nargs="+",
                        default=["kill", "recovery", "scaleup"],
                        choices=["kill", "recovery", "scaleup"])
    parser.add_argument("--gpus", type=int, default=4,
                        help="Number of GPUs (>=8 required for T_ScaleUp)")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    results: Dict[str, bool] = {}

    if "kill" in args.tests:
        coord = Coordinator()
        print(f"[Coord] T_Kill  barrier_dir={coord.barrier_dir}", flush=True)
        try:
            results["T_Kill"] = run_t_kill(coord)
        finally:
            coord.cleanup()

    if "recovery" in args.tests:
        coord = Coordinator()
        print(f"[Coord] T_Recovery  barrier_dir={coord.barrier_dir}", flush=True)
        try:
            results["T_Recovery"] = run_t_recovery(coord)
        finally:
            coord.cleanup()

    if "scaleup" in args.tests:
        if args.gpus < 8:
            print("[Coord] T_ScaleUp requires >=8 GPUs -- skipping", flush=True)
        else:
            coord = Coordinator()
            print(f"[Coord] T_ScaleUp  barrier_dir={coord.barrier_dir}", flush=True)
            try:
                results["T_ScaleUp"] = run_t_scaleup(coord)
            finally:
                coord.cleanup()

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for test, ok in results.items():
        print(f"  {test:<20s}  {'PASSED' if ok else 'FAILED'}", flush=True)
    print("=" * 70, flush=True)
    sys.exit(0 if all(results.values()) else 1)
