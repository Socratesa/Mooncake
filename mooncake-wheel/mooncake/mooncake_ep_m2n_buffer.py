import torch
import torch.distributed as dist
from typing import Callable, List, Optional, Tuple, Union

from mooncake.mooncake_ep_buffer import Buffer, EventOverlap


class M2NBuffer:
    """
    M2N (Attention-FFN disaggregation) wrapper around Mooncake EP Buffer.

    Handles cooperative dispatch/combine between attention ranks (token owners)
    and FFN ranks (expert owners). All expert IDs passed to this class must be
    PHYSICAL slot IDs — logical-to-physical mapping is the responsibility of
    the inference engine (e.g. SGLang EPLB).

    This class is a thin wrapper: it manages role distinction and constructs
    the correct empty/virtual tensors for each side, then delegates directly
    to Buffer. Return types are identical to Buffer.dispatch / Buffer.combine.

    active_ranks is NOT held internally — the inference engine owns and passes
    it each call, which is required for CUDA graph compatibility (the graph
    captures the tensor pointer; the engine updates values between replays).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        attention_ranks: List[int],
        ffn_ranks: List[int],
        num_experts_per_rank: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ep_buffer_bytes: Optional[int] = None,
    ):
        """
        Args:
            group: distributed process group containing all ranks.
            attention_ranks: list of rank ids that are attention instances.
            ffn_ranks: list of rank ids that are FFN instances.
            num_experts_per_rank: number of expert slots per rank (uniform).
            num_max_dispatch_tokens_per_rank: max tokens any rank dispatches.
            hidden: hidden dimension size.
            num_ep_buffer_bytes: EP buffer size in bytes. If None, computed
                                 automatically via get_ep_buffer_size_hint.
        """
        self.rank = group.rank()
        # Derive num_ranks from the rank lists rather than group.size().
        # group.size() may be stale after elastic scaling (e.g.
        # extend_group_size_to updates the backend but not PyTorch's cached
        # const Backend::size_).  The rank lists are always authoritative.
        self.num_ranks = len(attention_ranks) + len(ffn_ranks)

        self.attention_ranks = sorted(attention_ranks)
        self.ffn_ranks = sorted(ffn_ranks)
        assert set(self.attention_ranks) | set(self.ffn_ranks) == set(range(self.num_ranks)), \
            "attention_ranks + ffn_ranks must cover all ranks in the group"
        assert set(self.attention_ranks) & set(self.ffn_ranks) == set(), \
            "attention_ranks and ffn_ranks must not overlap"

        # Derive role from rank membership
        if self.rank in self.attention_ranks:
            self.role = 'attention'
        else:
            self.role = 'ffn'

        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.hidden = hidden
        self.num_experts_per_rank = num_experts_per_rank
        self.num_experts_total = self.num_ranks * num_experts_per_rank

        # Pre-compute attention rank virtual slot ranges for routing validation.
        # Slots on attention ranks have no expert weights — routing to them is
        # a bug in the inference engine's routing layer.
        if __debug__:
            self._attention_slots = self._build_attention_slots(
                self.attention_ranks, num_experts_per_rank,
            )

        # Create underlying EP buffer
        if num_ep_buffer_bytes is None:
            num_ep_buffer_bytes = Buffer.get_ep_buffer_size_hint(
                num_max_dispatch_tokens_per_rank, hidden,
                self.num_ranks, self.num_experts_total,
            )
        self.num_ep_buffer_bytes = num_ep_buffer_bytes
        self.buffer = Buffer(group, num_ep_buffer_bytes=num_ep_buffer_bytes)

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _build_attention_slots(attention_ranks: List[int], num_experts_per_rank: int) -> set:
        """Return the set of physical slot IDs that belong to attention ranks."""
        slots = set()
        for a_rank in attention_ranks:
            base = a_rank * num_experts_per_rank
            for j in range(num_experts_per_rank):
                slots.add(base + j)
        return slots

    def _validate_topk_idx(self, topk_idx: torch.Tensor) -> None:
        """
        Validate topk_idx values at dispatch/combine entry points.

        Checks:
        1. All values are -1 (padding) or in [0, num_experts_total).
        2. No value maps to an attention rank virtual slot.

        Only runs when __debug__ is True (i.e. not under python -O).
        Synchronizes GPU — do NOT use in latency-critical production paths
        without gating behind a flag.
        """
        if topk_idx.numel() == 0:
            return
        flat = topk_idx.view(-1)
        valid_mask = flat == -1
        valid_mask |= (flat >= 0) & (flat < self.num_experts_total)
        if not valid_mask.all().item():
            bad = flat[~valid_mask]
            raise ValueError(
                f"topk_idx contains out-of-bounds values: "
                f"{bad[:8].tolist()}... (expected -1 or [0, {self.num_experts_total}))"
            )

        # Check routing semantics: no token should target an attention rank slot
        if self._attention_slots:
            attn_mask = torch.zeros(
                self.num_experts_total, dtype=torch.bool, device=topk_idx.device,
            )
            attn_slot_ids = torch.tensor(
                sorted(self._attention_slots), dtype=torch.int64, device=topk_idx.device,
            )
            attn_mask[attn_slot_ids] = True

            real = flat[flat >= 0]
            if real.numel() > 0:
                hits_attn = attn_mask[real]
                if hits_attn.any().item():
                    bad_slots = real[hits_attn]
                    raise ValueError(
                        f"topk_idx routes to attention rank virtual slots "
                        f"(no expert weights): {bad_slots[:8].tolist()}... "
                        f"Attention slot range: {sorted(self._attention_slots)[:8]}..."
                    )

    # ── Dispatch (Attention → Expert) ────────────────────────────────

    def a2e_isend(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        active_ranks: torch.Tensor,
        timeout_us: int = -1,
        use_fp8: bool = True,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Optional[Callable],
    ]:
        """
        Dispatch tokens from an attention rank to FFN ranks.

        Args:
            x: [num_tokens, hidden] bfloat16, token hidden states.
            topk_idx: [num_tokens, num_topk] int64, PHYSICAL expert slot ids.
            active_ranks: [num_ranks] int32, 1=active 0=inactive.
            timeout_us: timeout in microseconds, -1 for no timeout.
            use_fp8: whether to use FP8 dispatch.
            async_finish: if True, operation may not be complete on return.
            return_recv_hook: if True, return a hook for deferred recv.

        Returns:
            recv_x: packed received data (or (data, scales) if use_fp8).
            packed_recv_count: [num_experts_per_rank] token counts.
            handle: opaque handle tuple for combine phase.
            event: EventOverlap for async synchronization.
            hook: recv hook callable, or None.
        """
        assert self.role == 'attention', \
            f"a2e_isend called on {self.role} rank"
        if __debug__ and not torch.cuda.is_current_stream_capturing():
            self._validate_topk_idx(topk_idx)
        return self.buffer.dispatch(
            x, topk_idx, active_ranks,
            self.num_max_dispatch_tokens_per_rank, self.num_experts_total,
            timeout_us, use_fp8=use_fp8,
            async_finish=async_finish, return_recv_hook=return_recv_hook,
        )

    def a2e_irecv(
        self,
        active_ranks: torch.Tensor,
        num_topk: int,
        timeout_us: int = -1,
        use_fp8: bool = True,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        torch.Tensor,
        Tuple,
        EventOverlap,
        Optional[Callable],
    ]:
        """
        Participate in dispatch from the FFN side (receive tokens).

        Args:
            active_ranks: [num_ranks] int32, 1=active 0=inactive.
            num_topk: number of top-k experts per token.
            timeout_us: timeout in microseconds, -1 for no timeout.
            use_fp8: whether to use FP8 dispatch.
            async_finish: if True, operation may not be complete on return.
            return_recv_hook: if True, return a hook for deferred recv.

        Returns:
            recv_x: packed received tokens for local experts.
            packed_recv_count: [num_experts_per_rank] token counts.
            handle: opaque handle tuple for combine phase.
            event: EventOverlap for async synchronization.
            hook: recv hook callable, or None.
        """
        assert self.role == 'ffn', \
            f"a2e_irecv called on {self.role} rank"
        empty_x = torch.empty(0, self.hidden, dtype=torch.bfloat16, device='cuda')
        empty_topk = torch.empty(0, num_topk, dtype=torch.int64, device='cuda')
        return self.buffer.dispatch(
            empty_x, empty_topk, active_ranks,
            self.num_max_dispatch_tokens_per_rank, self.num_experts_total,
            timeout_us, use_fp8=use_fp8,
            async_finish=async_finish, return_recv_hook=return_recv_hook,
        )

    # ── Combine (Expert → Attention) ─────────────────────────────────

    def e2a_irecv(
        self,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        active_ranks: torch.Tensor,
        handle: Tuple,
        timeout_us: int = -1,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Optional[Callable]]:
        """
        Combine expert outputs back to this attention rank.

        Args:
            topk_idx: [num_tokens, num_topk] int64, same as dispatch phase.
            topk_weights: [num_tokens, num_topk] float32, gating weights.
            active_ranks: [num_ranks] int32, 1=active 0=inactive.
            handle: opaque handle returned from a2e_isend dispatch phase.
            timeout_us: timeout in microseconds, -1 for no timeout.
            async_finish: if True, operation may not be complete on return.
            return_recv_hook: if True, return a hook for deferred recv.
            out: optional pre-allocated output tensor.

        Returns:
            combined_x: [num_tokens, hidden] bfloat16, weighted sum.
            event: EventOverlap for async synchronization.
            hook: recv hook callable, or None.
        """
        assert self.role == 'attention', \
            f"e2a_irecv called on {self.role} rank"
        if __debug__ and not torch.cuda.is_current_stream_capturing():
            self._validate_topk_idx(topk_idx)
        virtual_x = torch.zeros(
            self.num_experts_per_rank,
            self.num_ranks * self.num_max_dispatch_tokens_per_rank,
            self.hidden,
            dtype=torch.bfloat16, device='cuda',
        )
        return self.buffer.combine(
            virtual_x, topk_idx, topk_weights, active_ranks,
            timeout_us, handle,
            async_finish=async_finish, return_recv_hook=return_recv_hook,
            out=out,
        )

    def e2a_isend(
        self,
        x: torch.Tensor,
        active_ranks: torch.Tensor,
        handle: Tuple,
        timeout_us: int = -1,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[torch.Tensor, EventOverlap, Optional[Callable]]:
        """
        Send expert outputs back from this FFN rank.

        Args:
            x: [num_experts_per_rank, num_ranks * max_tokens, hidden] bfloat16,
               expert computation results.
            active_ranks: [num_ranks] int32, 1=active 0=inactive.
            handle: opaque handle returned from a2e_irecv dispatch phase.
            timeout_us: timeout in microseconds, -1 for no timeout.
            zero_copy: if True, skip extra copy (x must come from
                       get_next_combine_buffer).
            async_finish: if True, operation may not be complete on return.
            return_recv_hook: if True, return a hook for deferred recv.

        Returns:
            combined_x: empty tensor (FFN rank has no tokens to combine).
            event: EventOverlap for async synchronization.
            hook: recv hook callable, or None.
        """
        assert self.role == 'ffn', \
            f"e2a_isend called on {self.role} rank"
        empty_topk = torch.empty(0, 1, dtype=torch.int64, device='cuda')
        empty_weights = torch.empty(0, 1, dtype=torch.float32, device='cuda')
        return self.buffer.combine(
            x, empty_topk, empty_weights, active_ranks,
            timeout_us, handle,
            zero_copy=zero_copy, async_finish=async_finish,
            return_recv_hook=return_recv_hook,
        )

    # ── Utility ──────────────────────────────────────────────────────

    def get_next_combine_buffer(self, handle: Tuple) -> torch.Tensor:
        """
        Get pre-allocated buffer for zero-copy combine.

        The FFN rank writes expert outputs into this buffer, then passes it
        to e2a_isend with zero_copy=True.

        Args:
            handle: opaque handle returned from a2e_irecv dispatch phase.

        Returns:
            buffer: [num_experts_per_rank, num_ranks * max_tokens, hidden]
                    bfloat16 tensor.
        """
        return self.buffer.get_next_combine_buffer(handle)

    def update_ep_member(self):
        """
        Re-exchange RDMA metadata after a rank recovers.

        Call this on ALL ranks (collective) after a previously-failed rank
        comes back online. The inference engine should set
        active_ranks[recovered_rank] = 1 before the next dispatch/combine.

        This does NOT rebuild the Buffer — use scale_up() when the group
        topology (number of ranks) changes.
        """
        self.buffer.update_ep_member()

    # ── Scaling (topology change) ────────────────────────────────────

    def scale_up(
        self,
        new_group: dist.ProcessGroup,
        new_attention_ranks: List[int],
        new_ffn_ranks: List[int],
        new_num_experts_per_rank: int,
        num_ep_buffer_bytes: Optional[int] = None,
    ):
        """
        Rebuild M2NBuffer after a topology change (group_size changed).

        Only needed when ranks are added/removed from the group. For recovery
        of existing ranks (same group_size), use update_ep_member() instead.

        Args:
            new_group: new process group with all ranks.
            new_attention_ranks: updated attention rank list.
            new_ffn_ranks: updated FFN rank list.
            new_num_experts_per_rank: new expert slots per rank.
            num_ep_buffer_bytes: new buffer size. If None, auto-computed.
        """
        new_num_ranks = len(new_attention_ranks) + len(new_ffn_ranks)
        new_rank = new_group.rank()

        self.rank = new_rank
        self.num_ranks = new_num_ranks
        self.attention_ranks = sorted(new_attention_ranks)
        self.ffn_ranks = sorted(new_ffn_ranks)

        # Re-derive role
        if self.rank in self.attention_ranks:
            self.role = 'attention'
        else:
            self.role = 'ffn'

        self.num_experts_per_rank = new_num_experts_per_rank
        self.num_experts_total = new_num_ranks * new_num_experts_per_rank

        # Rebuild underlying buffer (group_size changed)
        if num_ep_buffer_bytes is None:
            num_ep_buffer_bytes = Buffer.get_ep_buffer_size_hint(
                self.num_max_dispatch_tokens_per_rank, self.hidden,
                self.num_ranks, self.num_experts_total,
            )
        self.num_ep_buffer_bytes = num_ep_buffer_bytes
        self.buffer = Buffer(new_group, num_ep_buffer_bytes=num_ep_buffer_bytes)
