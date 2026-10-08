"""Blockwise BF16 historical memory and configurable exact-local lifecycle.

Fine states seal only after the serve limit has served its last query.
The oldest fine block becomes one BF16 fine; local_keep exact tokens remain.
Training uses differentiable state tensors.  Inference retains only sealed
fine/page states and exact local K/V, with an extra raw local K needed to seal.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .indirect_fine import selected_fine_numerator
from .selected_read import selected_fine_read
from .routing_scores import candidate_scores
from .config import TaoConfig
from .structured_writer import structured_statistics, structured_matrix
from .association_writer import association_statistics, association_matrix, chronological_sum, feature_read_coefficients


DEFAULT_CONFIG = TaoConfig()
FINE_LENGTH = DEFAULT_CONFIG.fine_tokens


def _joint_budget(count, config=DEFAULT_CONFIG):
    return config.budget(count)


def routed_counts(completed_pages, pending_fine, config=DEFAULT_CONFIG):
    """Return completed-page K, eligible fine U, and global selected fine M."""
    group = config.coarse_group
    coarse_budget = _joint_budget(group * completed_pages, config)
    coarse = min(completed_pages,
                 (2 * coarse_budget + group - 1) // group)
    all_visible = group * completed_pages + pending_fine
    eligible = group * coarse + pending_fine
    return coarse, eligible, min(eligible, _joint_budget(all_visible, config))


def inference_routed_counts(completed_pages, pending_fine, config=DEFAULT_CONFIG):
    """Inference changes route timing, not the joint count policy."""
    return routed_counts(completed_pages, pending_fine, config)


def _einsum(equation, *values):
    with torch.autocast(device_type=values[0].device.type, enabled=False):
        return torch.einsum(equation, *values)


def _topk_oldest(scores, count):
    """Chronological first occurrence wins exact ties."""
    return scores.argsort(dim=-1, descending=True, stable=True)[
        ..., :min(count, scores.shape[-1])]


def _seal(k, v, config=DEFAULT_CONFIG):
    """Return selected BF16 descriptor/C and optional true FP32 statistics."""
    if config.fine_writer == "structured_gram_association":
        statistics = structured_statistics(k, v)
        matrix = structured_matrix(*statistics, config.fine_tokens)
        return statistics[2].log().to(k.dtype), matrix, statistics
    if config.fine_writer in ("feature_mass_association", "structured_gram_association"):
        statistics = association_statistics(k, v, with_mass=True)
        matrix = association_matrix(*statistics[:2], unchecked=True)
        return statistics[2].log().to(k.dtype), matrix, statistics
    maximum = k.amax(dim=1, keepdim=True)
    exponent = (k - maximum).exp().to(k.dtype)
    partition = exponent.sum(dim=1, keepdim=True).to(k.dtype)
    probability = (exponent / partition).to(k.dtype)
    z = (maximum.squeeze(1) + partition.squeeze(1).log()).to(k.dtype)
    statistics = None
    if config.fine_writer == "regularized_association":
        statistics = association_statistics(k, v)
        matrix = association_matrix(*statistics)
    else:
        matrix = _einsum("bthd,bthe->bhde", probability, v).to(k.dtype)
    return z, matrix, statistics


def _seal_training_all(k, v, config=DEFAULT_CONFIG):
    """Seal independent fine-token BF16 segments in one differentiable batch."""
    b, total, heads, d = k.shape
    fine = config.fine_tokens
    segments = total // fine - 1
    keys = k[:, :segments * fine].reshape(b, segments, fine, heads, d)
    values = v[:, :segments * fine].reshape(b, segments, fine, heads, d)
    if config.fine_writer == "structured_gram_association":
        square, cross, mass = structured_statistics(
            keys.reshape(b * segments, fine, heads, d),
            values.reshape(b * segments, fine, heads, d))
        matrix = structured_matrix(square, cross, mass, fine)
        matrix = matrix.reshape(b, segments, heads, d, d).permute(0, 2, 1, 3, 4)
        square = square.reshape(b, segments, heads, d).permute(0, 2, 1, 3)
        cross = cross.reshape(b, segments, heads, d, d).permute(0, 2, 1, 3, 4)
        mass = mass.reshape(b, segments, heads, d).permute(0, 2, 1, 3)
        return mass.log().to(k.dtype), matrix, (square, cross, mass)
    if config.fine_writer in ("feature_mass_association", "structured_gram_association"):
        gram, cross, mass = association_statistics(
            keys.reshape(b * segments, fine, heads, d),
            values.reshape(b * segments, fine, heads, d), with_mass=True)
        matrix = association_matrix(gram, cross, unchecked=True)
        matrix = matrix.reshape(b, segments, heads, d, d).permute(0, 2, 1, 3, 4)
        statistics = tuple(t.reshape(b, segments, heads, d, d).permute(0, 2, 1, 3, 4)
                           for t in (gram, cross))
        mass = mass.reshape(b, segments, heads, d).permute(0, 2, 1, 3)
        return mass.log().to(k.dtype), matrix, (*statistics, mass)
    maximum = keys.amax(dim=2, keepdim=True)
    exponent = (keys - maximum).exp().to(k.dtype)
    partition = exponent.sum(dim=2, keepdim=True).to(k.dtype)
    probability = (exponent / partition).to(k.dtype)
    z = (maximum.squeeze(2) + partition.squeeze(2).log()).to(k.dtype)
    statistics = None
    if config.fine_writer == "regularized_association":
        gram, cross = association_statistics(
            keys.reshape(b * segments, fine, heads, d),
            values.reshape(b * segments, fine, heads, d))
        matrix = association_matrix(gram, cross)
        matrix = matrix.reshape(b, segments, heads, d, d).permute(0, 2, 1, 3, 4)
        statistics = tuple(t.reshape(b, segments, heads, d, d).permute(0, 2, 1, 3, 4)
                           for t in (gram, cross))
    else:
        matrix = _einsum("bsihd,bsihe->bhsde", probability, values).to(k.dtype)
    return z.permute(0, 2, 1, 3), matrix, statistics


class LayerState:
    """Ordered completed coarse pages and sealed fine in the pending page."""

    def __init__(self, packed=False, progressive_layer=None,
                 pending_coarse_elision=False, fused_fine=False,
                 config=DEFAULT_CONFIG):
        self.config = config
        self.pages = []  # (page_z, page_C, fine_z, fine_C)
        self.pending = []  # (fine_z, fine_C)
        self.pending_gram = None
        self.pending_cross = None
        self.pending_mass = None
        self.pending_square_mass = None
        self.packed = packed
        self.fused_fine = fused_fine
        self.fused_fine_calls = 0
        self.history_region = None
        self.packed_fine_c = None
        self.fine_matrix_repack_bytes = 0
        self.state_version = 0
        self.page_normalization_memo = None
        self.page_memo_hits = 0
        self.page_memo_misses = 0
        self.page_memo_invalidations = 0
        self.route_key = None
        self.route_ids = None  # Integer page/fine IDs only; never weights or C views.
        self.fine_observer = None  # Post-read diagnostic only; never feeds B into A.
        self.last_route_queries = 0
        self.progressive_layer = progressive_layer
        self.pending_coarse_elision = pending_coarse_elision
        self.pending_coarse_constructions = 0
        self.pending_coarse_elision_reads = 0
        self.last_fine_contractions = 0
        self.route_refreshes = 0
        self.peak_selected_fine = 0

    def read(self, q, position=0, route_interval=1, allow_page_memo=False):
        self.last_route_queries = 0
        self.last_fine_contractions = 0
        if self.progressive_layer is not None:
            return self._read_progressive(q, position, allow_page_memo)
        if route_interval == 1:
            if self.pages or self.pending:
                self.last_route_queries = q.shape[0] * q.shape[1] * q.shape[2]
            return self._read_per_query(q)
        if not self.pages and not self.pending:
            empty = torch.zeros_like(q)
            return empty, empty, False, False
        coarse, fine = [], []
        offset = 0
        while offset < q.shape[1]:
            absolute = position + offset  # Zero-based query position.
            block = (absolute // self.config.fine_tokens) * self.config.fine_tokens
            anchor = block + ((absolute - block) // route_interval) * route_interval
            count = min(q.shape[1] - offset, anchor + route_interval - absolute,
                        block + self.config.fine_tokens - absolute)
            key = (self.state_version, anchor, route_interval)
            refresh = self.route_key != key
            if refresh and absolute != anchor:
                raise ValueError("route anchor is absent from this sequence's stream cache")
            c, f = self._read_reused(q[:, offset:offset + count], refresh)
            if refresh:
                self.route_key = key
                self.last_route_queries += q.shape[0] * q.shape[2]
            coarse.append(c)
            fine.append(f)
            offset += count
        return (torch.cat(coarse, dim=1), torch.cat(fine, dim=1),
                bool(self.pages) and not self.config.coarse_index_only, True)

    def _read_progressive(self, q, position, allow_page_memo=False):
        if not self.pages and not self.pending:
            empty = torch.zeros_like(q)
            return empty, empty, False, False
        coarse, fine = [], []
        offset = 0
        first_fine = self.config.first_fine
        first_turn = first_fine + self.config.layer_stagger * self.progressive_layer
        period = self.config.route_period
        while offset < q.shape[1]:
            absolute = position + offset
            refresh = (absolute == first_fine or
                       (absolute >= first_turn and (absolute - first_turn) % period == 0))
            next_turn = (first_turn if absolute < first_turn else
                         first_turn + period * ((absolute - first_turn) // period + 1))
            count = min(q.shape[1] - offset, next_turn - absolute)
            c, f = self._read_reused(q[:, offset:offset + count], refresh,
                                     allow_page_memo=allow_page_memo and q.shape[1] == 1,
                                     allow_region=allow_page_memo and q.shape[1] == 1)
            if refresh:
                self.last_route_queries += q.shape[0] * q.shape[2]
                self.route_refreshes += 1
            selected = self.route_ids[1].shape[-1]
            self.last_fine_contractions += q.shape[0] * count * q.shape[2] * selected
            self.peak_selected_fine = max(self.peak_selected_fine, selected)
            coarse.append(c)
            fine.append(f)
            offset += count
        if q.shape[1] == 1:
            return coarse[0], fine[0], bool(self.pages) and not self.config.coarse_index_only, True
        return (torch.cat(coarse, dim=1), torch.cat(fine, dim=1),
                bool(self.pages) and not self.config.coarse_index_only, True)

    def _read_per_query(self, q):
        outputs = [self._read_reused(q[:, i:i + 1], True)
                   for i in range(q.shape[1])]
        coarse, fine = zip(*outputs)
        return (torch.cat(coarse, dim=1), torch.cat(fine, dim=1),
                bool(self.pages) and not self.config.coarse_index_only, bool(self.pages or self.pending))

    def _read_reused(self, q, refresh, allow_page_memo=False, allow_region=False):
        # Both execution modes call the same memo-backed arithmetic region.
        # Memo publication/counters stay outside capture and use real queries.
        if (allow_page_memo and q.shape[1] == 1 and not refresh
                and self.config.q1_page_normalization_memo
                and self.config.coarse_index_only and self.config.fine_writer == "softmax"
                and self.fused_fine and self.packed):
            self._prepare_q1_page_memo(q.dtype)
            self.fused_fine_calls += 1
            if self.config.q1_history_region_graph and allow_region:
                if self.history_region is None:
                    from .history_region_graph import HistoryRegionGraph
                    self.history_region = HistoryRegionGraph()
                return self.history_region.run(self, q)
            return self._q1_page_memo_region(q, self.route_ids[1])
        b, length, heads, d = q.shape
        dtype = q.dtype
        h = F.softmax(q.float() if self.config.fine_writer in ("feature_mass_association", "structured_gram_association") else q,
                      dim=-1).to(dtype)
        memo_eligible = (allow_page_memo and length == 1 and not refresh
                         and self.config.q1_page_normalization_memo
                         and self.config.coarse_index_only
                         and self.config.fine_writer == "softmax")
        memo = self.page_normalization_memo if memo_eligible else None
        if memo is not None and memo[0] != self.state_version:
            memo = None
        if memo_eligible:
            if memo is None:
                self.page_memo_misses += 1
            else:
                self.page_memo_hits += 1
        page_z, page_c = [], []
        fine_z_parts, fine_c_parts = [], []
        pending_z = None
        for pz, pc, fz, fc in self.pages:
            page_z.append(pz)
            page_c.append(pc)
            fine_z_parts.append(fz)
            if not self.packed:
                fine_c_parts.append(fc)

        if self.pending:
            pending_z = (memo[1] if memo is not None else
                         torch.stack([z for z, _ in self.pending], dim=2))
            if memo is None:
                pz = (self.pending_mass.log().to(dtype)
                      if self.config.fine_writer in ("feature_mass_association", "structured_gram_association") else
                      torch.logsumexp(pending_z, dim=2).to(dtype))
                page_z.append(pz)
            fine_z_parts.append(pending_z)
            if not self.packed:
                fine_c_parts.append(torch.stack([c for _, c in self.pending], dim=2))

        if memo is None:
            page_z = torch.stack(page_z, dim=2)
            page_weight = F.softmax(page_z, dim=2).to(dtype)
            if memo_eligible:
                # Immutable derived operands for this seal epoch; never query weights.
                self.page_normalization_memo = (self.state_version, pending_z,
                                                page_z, page_weight)
        else:
            page_z, page_weight = memo[2], memo[3]
        hv = (h.unsqueeze(3) * page_weight[:, None]).to(dtype)
        if self.pages and not self.config.coarse_index_only:
            page_c = torch.stack(page_c, dim=2)
            coarse_weight = F.softmax(page_z[:, :, :len(self.pages)], dim=2).to(dtype)
            coarse_hv = (feature_read_coefficients(h, page_z[:, None, :, :len(self.pages)])
                         if self.config.fine_writer in ("feature_mass_association", "structured_gram_association") else
                         (h.unsqueeze(3) * coarse_weight[:, None]).to(dtype))
            coarse = _einsum("bthpd,bhpde->bthe", coarse_hv, page_c).to(dtype)
        else:
            coarse = torch.zeros_like(q)

        fine_z = torch.cat(fine_z_parts, dim=2)
        if refresh:
            # Only the first query is the absolute anchor. Later queries retain
            # their own h/hv and recompute selected channel weights below.
            full_hv, full_length = hv, length
            hv, length = hv[:, :1], 1
            completed = len(self.pages)
            pending_count = len(self.pending)
            coarse_count, _, fine_count = inference_routed_counts(
                completed, pending_count, self.config)
            candidate_pages = []
            candidate_ids = []
            if coarse_count:
                page_mass = hv[..., :completed, :].sum(-1)
                chosen_page = _topk_oldest(page_mass, coarse_count).sort(-1).values
                segment = torch.arange(self.config.coarse_group, device=q.device)
                repeated_pages = chosen_page.unsqueeze(-1).expand(
                    b, length, heads, coarse_count, self.config.coarse_group).reshape(
                        b, length, heads, coarse_count * self.config.coarse_group)
                candidate_pages.append(repeated_pages)
                candidate_ids.append((chosen_page.unsqueeze(-1) * self.config.coarse_group
                                      + segment).reshape(
                                          b, length, heads,
                                          coarse_count * self.config.coarse_group))
            if pending_count:
                pending_ids = torch.arange(pending_count, device=q.device)
                pending_ids = pending_ids.view(1, 1, 1, pending_count).expand(
                    b, length, heads, pending_count)
                candidate_pages.append(torch.full_like(pending_ids, completed))
                candidate_ids.append(completed * self.config.coarse_group + pending_ids)
            candidate_pages = torch.cat(candidate_pages, dim=-1)
            candidate_ids = torch.cat(candidate_ids, dim=-1)

            fine_z = torch.cat(fine_z_parts, dim=2)
            z_candidates = torch.gather(
                fine_z[:, None].expand(-1, length, -1, -1, -1), 3,
                candidate_ids.unsqueeze(-1).expand(-1, -1, -1, -1, d))
            page_z_candidates = torch.gather(
                page_z[:, None].expand(-1, length, -1, -1, -1), 3,
                candidate_pages.unsqueeze(-1).expand(-1, -1, -1, -1, d))
            segment_weight = (z_candidates - page_z_candidates).exp().to(dtype)
            page_channel = torch.gather(
                hv, 3, candidate_pages.unsqueeze(-1).expand(-1, -1, -1, -1, d))
            channel_mass = (page_channel * segment_weight).to(dtype)
            mass = channel_mass.sum(-1)
            selected = _topk_oldest(mass, fine_count)
            selected_id = candidate_ids.gather(-1, selected)

            self.route_ids = (chosen_page if coarse_count else candidate_pages[..., :0], selected_id)
            if self.history_region is not None:
                self.history_region.update_routes(selected_id)
            hv, length = full_hv, full_length
        selected_id = self.route_ids[1].expand(-1, length, -1, -1)
        if self.config.fine_writer in ("feature_mass_association", "structured_gram_association"):
            selected_z = fine_z[:, None].expand(-1, length, -1, -1, -1).gather(
                3, selected_id.unsqueeze(-1).expand(-1, -1, -1, -1, d))
            channel = feature_read_coefficients(h, selected_z)
            bank = self.packed_fine_c if self.packed else torch.cat(fine_c_parts, dim=2)
            return coarse, selected_fine_numerator(channel, selected_id, bank).to(dtype)
        if self.fused_fine:
            from .query_fresh_fine import query_fresh_fine_read
            self.fused_fine_calls += 1
            fine = query_fresh_fine_read(
                hv, page_z, fine_z, selected_id, self.packed_fine_c,
                self.config.coarse_group)
            return coarse, fine
        selected_page = selected_id // self.config.coarse_group
        z_selected = fine_z[:, None].expand(-1, length, -1, -1, -1).gather(
            3, selected_id.unsqueeze(-1).expand(-1, -1, -1, -1, d))
        pz_selected = page_z[:, None].expand(-1, length, -1, -1, -1).gather(
            3, selected_page.unsqueeze(-1).expand(-1, -1, -1, -1, d))
        segment_weight = (z_selected - pz_selected).exp().to(dtype)
        page_channel = hv.gather(
            3, selected_page.unsqueeze(-1).expand(-1, -1, -1, -1, d))
        selected_channel = (page_channel * segment_weight).to(dtype)
        fine_c = (self.packed_fine_c if self.packed else torch.cat(fine_c_parts, dim=2))
        numerator = selected_fine_numerator(selected_channel, selected_id, fine_c).to(dtype)
        denominator = selected_channel.sum(-1).sum(-1, keepdim=True).to(dtype)
        fine = (numerator / denominator).to(dtype)
        if self.fine_observer is not None:
            self.fine_observer(
                layer=self.progressive_layer, refresh=refresh,
                hv=hv, page_z=page_z, fine_z=fine_z,
                selected_id=selected_id, fine_c=fine_c,
                selected_page=selected_page, z_selected=z_selected,
                pz_selected=pz_selected, segment_weight=segment_weight,
                page_channel=page_channel, selected_channel=selected_channel,
                numerator=numerator, denominator=denominator, fine=fine)
        return coarse, fine

    def _prepare_q1_page_memo(self, dtype):
        """Original seal-epoch memo preparation, never executed in capture."""
        memo = self.page_normalization_memo
        if memo is not None and memo[0] == self.state_version:
            self.page_memo_hits += 1
            return
        if self.history_region is not None:
            self.history_region.invalidate("memo_replace")
        self.page_memo_misses += 1
        page_z = [pz for pz, _, _, _ in self.pages]
        pending_z = None
        if self.pending:
            pending_z = torch.stack([z for z, _ in self.pending], dim=2)
            page_z.append(torch.logsumexp(pending_z, dim=2).to(dtype))
        page_z = torch.stack(page_z, dim=2)
        page_weight = F.softmax(page_z, dim=2).to(dtype)
        self.page_normalization_memo = (self.state_version, pending_z,
                                        page_z, page_weight)

    def _q1_page_memo_region(self, q, selected_ids):
        """One pure arithmetic path for eager calls and graph capture/replay."""
        dtype = q.dtype
        h = F.softmax(q, dim=-1).to(dtype)
        _, pending_z, page_z, page_weight = self.page_normalization_memo
        hv = (h.unsqueeze(3) * page_weight[:, None]).to(dtype)
        coarse = torch.zeros_like(q)
        fine_z_parts = [fz for _, _, fz, _ in self.pages]
        if pending_z is not None:
            fine_z_parts.append(pending_z)
        fine_z = torch.cat(fine_z_parts, dim=2)
        from .query_fresh_fine import query_fresh_fine_read
        fine = query_fresh_fine_read(hv, page_z, fine_z,
            selected_ids.expand(-1, q.shape[1], -1, -1), self.packed_fine_c,
            self.config.coarse_group)
        return coarse, fine

    def seal(self, old_k, old_v):
        """Publish one entire old local block after its final causal query."""
        if old_k.shape[1] != self.config.fine_tokens or old_v.shape != old_k.shape:
            raise ValueError("fine seal needs exactly one fine_tokens old local block")
        fine_z, fine_c, statistics = _seal(old_k, old_v, self.config)
        self.append_fine(fine_z, fine_c, statistics)

    def append_fine(self, fine_z, fine_c, statistics=None):
        """Publish an already constructed fine at its causal seal boundary."""
        if self.history_region is not None:
            self.history_region.invalidate("seal")
        # Publication occurs after the current read; queued current-stream users
        # retain allocator safety while the request releases its old epoch.
        if self.page_normalization_memo is not None:
            self.page_normalization_memo = None
            self.page_memo_invalidations += 1
        self.state_version += 1
        if self.progressive_layer is None:
            self.route_key = None
            self.route_ids = None
        if self.packed:
            if self.packed_fine_c is None:
                self.packed_fine_c = fine_c.unsqueeze(2)
            else:
                self.packed_fine_c = torch.cat(
                    (self.packed_fine_c, fine_c.unsqueeze(2)), dim=2)
                self.fine_matrix_repack_bytes += (
                    self.packed_fine_c.numel() * self.packed_fine_c.element_size())
        if self.config.fine_writer in ("regularized_association", "feature_mass_association", "structured_gram_association"):
            gram, cross = statistics[:2]
            if self.config.fine_writer in ("feature_mass_association", "structured_gram_association"):
                mass = statistics[2]
                self.pending_mass = (mass if self.pending_mass is None else self.pending_mass + mass)
            if self.config.fine_writer == "structured_gram_association":
                self.pending_square_mass = (gram if self.pending_square_mass is None else
                                            self.pending_square_mass + gram)
                self.pending_cross = cross if self.pending_cross is None else self.pending_cross + cross
            elif self.pending_gram is None:
                self.pending_gram, self.pending_cross = gram, cross
            else:
                self.pending_gram = self.pending_gram + gram
                self.pending_cross = self.pending_cross + cross
        self.pending.append((fine_z, fine_c))
        if len(self.pending) == self.config.coarse_group:
            z = torch.stack([pair[0] for pair in self.pending], dim=2)
            c = (self.packed_fine_c[:, :, -self.config.coarse_group:]
                 if self.config.coarse_index_only and self.packed else
                 torch.stack([pair[1] for pair in self.pending], dim=2))
            page_z = (self.pending_mass.log().to(z.dtype)
                      if self.config.fine_writer in ("feature_mass_association", "structured_gram_association") else
                      torch.logsumexp(z, dim=2).to(z.dtype))
            if not self.config.coarse_index_only:
                weight = (z - page_z.unsqueeze(2)).exp().to(z.dtype)
            if self.config.coarse_index_only:
                page_c = None
            elif self.config.fine_writer in ("regularized_association", "feature_mass_association", "structured_gram_association"):
                if self.config.fine_writer == "structured_gram_association":
                    page_c = structured_matrix(self.pending_square_mass, self.pending_cross,
                        self.pending_mass, self.config.fine_tokens * len(self.pending))
                else:
                    page_c = association_matrix(self.pending_gram, self.pending_cross,
                        unchecked=self.config.fine_writer == "feature_mass_association")
                self.pending_gram = self.pending_cross = self.pending_mass = self.pending_square_mass = None
            else:
                page_c = _einsum("bhsd,bhsde->bhde", weight, c).to(z.dtype)
            self.pages.append((page_z, page_c, z, c))
            self.pending.clear()
        if self.packed:
            # All persistent fine matrices become views of one owned buffer.
            # Repacking occurs at a seal, never during each query-block read.
            self.pages = [
                (pz, pc, fz, self.packed_fine_c[:, :, i * self.config.coarse_group:
                                                (i + 1) * self.config.coarse_group])
                for i, (pz, pc, fz, _) in enumerate(self.pages)]
            offset = len(self.pages) * self.config.coarse_group
            self.pending = [
                (z, self.packed_fine_c[:, :, offset + i])
                for i, (z, _) in enumerate(self.pending)]

    def pending_writer_bytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (self.pending_gram, self.pending_cross, self.pending_mass, self.pending_square_mass) if t is not None)

    def logical_bytes(self):
        count = len(self.pages) * (self.config.coarse_group + 1) + len(self.pending)
        if not count:
            return 0
        sample = self.pages[0][0] if self.pages else self.pending[0][0]
        b, h, d = sample.shape
        elements = count * (d * d + d)
        if self.config.coarse_index_only:
            elements -= len(self.pages) * d * d
        return elements * b * h * sample.element_size()


def _training_metadata(fine_z, fine_c, config=DEFAULT_CONFIG, statistics=None):
    """Build pending z prefixes; construct C only for completed pages."""
    total = fine_z.shape[2]
    summaries = {}
    group = config.coarse_group
    for n in range(1, min(group, total) + 1):
        pages = (total - n) // group + 1
        indices = (torch.arange(pages, device=fine_z.device)[:, None]
                   * group
                   + torch.arange(n, device=fine_z.device)[None, :])
        # One index-select backward instead of leading basic-slice backwards.
        z = fine_z.index_select(2, indices.flatten()).reshape(
            *fine_z.shape[:2], pages, n, fine_z.shape[-1])
        if config.fine_writer in ("feature_mass_association", "structured_gram_association"):
            mass = statistics[2].index_select(2, indices.flatten()).reshape(
                *fine_z.shape[:2], pages, n, fine_z.shape[-1])
            total_mass = mass.select(3, 0)
            for index in range(1, n):
                total_mass = total_mass + mass.select(3, index)
            pz = total_mass.log().to(z.dtype)
        else:
            pz = torch.logsumexp(z, dim=3).to(z.dtype)
        if n == group and not config.coarse_index_only:
            c = fine_c.index_select(2, indices.flatten()).reshape(
                *fine_c.shape[:2], pages, n, *fine_c.shape[-2:])
            weight = (z - pz.unsqueeze(3)).exp().to(z.dtype)
            if config.fine_writer == "structured_gram_association":
                square = statistics[0].index_select(2, indices.flatten()).reshape(
                    *fine_z.shape[:2], pages, n, fine_z.shape[-1])
                total_square = square.select(3, 0)
                for index in range(1, n):
                    total_square = total_square + square.select(3, index)
                cross = statistics[1].index_select(2, indices.flatten()).reshape(
                    *fine_c.shape[:2], pages, n, *fine_c.shape[-2:])
                pc = structured_matrix(total_square, chronological_sum(cross),
                                       total_mass, n * config.fine_tokens)
            elif config.fine_writer in ("regularized_association", "feature_mass_association"):
                grouped = [t.index_select(2, indices.flatten()).reshape(
                    *t.shape[:2], pages, n, *t.shape[-2:]) for t in statistics[:2]]
                pc = association_matrix(*(chronological_sum(t) for t in grouped),
                    unchecked=config.fine_writer in ("feature_mass_association", "structured_gram_association"))
            else:
                pc = _einsum("bhpsd,bhpsde->bhpde", weight, c).to(z.dtype)
            summaries[n] = (pz.unbind(2), pc.unbind(2))
        else:
            summaries[n] = (pz.unbind(2), None)
    return summaries


def _tile_metadata(summaries, fine_z, visible, config=DEFAULT_CONFIG):
    completed, pending = divmod(visible, config.coarse_group)
    pz, pc = [], []
    if completed:
        full_z, full_c = summaries[config.coarse_group]
        pz.extend(full_z[:completed])
        if full_c is not None:
            pc.extend(full_c[:completed])
    if pending:
        partial_z, _ = summaries[pending]
        pz.append(partial_z[completed])
    coarse_c = (torch.stack(pc, dim=2) if pc else
                fine_z.new_empty((*fine_z.shape[:2], 0,
                                  fine_z.shape[-1], fine_z.shape[-1])))
    return torch.stack(pz, dim=2), coarse_c, fine_z.narrow(2, 0, visible)


def _read_training_group(q, page_z, page_c, fine_z, fine_c,
                         visible_fine, completed, pending_width, fine_count,
                         route_interval=1, config=DEFAULT_CONFIG):
    """Independent causal tiles packed on the batch axis, never the time axis.

    All members have identical completed-page count, pending-page presence,
    and M. Only the pending fine axis is padded. Its z is -inf before scores,
    and its selection score is -inf; no padded/future C can be selected.
    """
    b, length, heads, d = q.shape
    dtype = q.dtype
    h = F.softmax(q.float() if config.fine_writer in ("feature_mass_association", "structured_gram_association") else q,
                  dim=-1).to(dtype)
    page_weight = F.softmax(page_z, dim=2).to(dtype)
    hv = (h.unsqueeze(3) * page_weight[:, None]).to(dtype)
    if completed and not config.coarse_index_only:
        coarse_weight = F.softmax(page_z[:, :, :completed], dim=2).to(dtype)
        coarse_hv = (feature_read_coefficients(h, page_z[:, None, :, :completed])
                     if config.fine_writer in ("feature_mass_association", "structured_gram_association") else
                     (h.unsqueeze(3) * coarse_weight[:, None]).to(dtype))
        coarse = _einsum("bthpd,bhpde->bthe", coarse_hv, page_c).to(dtype)
    else:
        coarse = torch.zeros_like(q)
    route_hv = hv if route_interval == 1 else hv[:, ::route_interval]
    route_length = route_hv.shape[1]
    coarse_count, _, _ = routed_counts(completed, pending_width, config)
    candidate_pages, candidate_ids = [], []
    if coarse_count:
        page_mass = route_hv[..., :completed, :].sum(-1)
        chosen_page = _topk_oldest(page_mass, coarse_count).sort(-1).values
        segment = torch.arange(config.coarse_group, device=q.device)
        candidate_pages.append(chosen_page.unsqueeze(-1).expand(
            b, route_length, heads, coarse_count, config.coarse_group).reshape(
                b, route_length, heads, coarse_count * config.coarse_group))
        candidate_ids.append((chosen_page.unsqueeze(-1) * config.coarse_group
                              + segment).reshape(
                                  b, route_length, heads,
                                  coarse_count * config.coarse_group))
    if pending_width:
        pending_ids = torch.arange(pending_width, device=q.device)
        pending_ids = pending_ids.view(1, 1, 1, pending_width).expand(
            b, route_length, heads, pending_width)
        candidate_pages.append(torch.full_like(pending_ids, completed))
        candidate_ids.append(completed * config.coarse_group + pending_ids)
    candidate_pages = torch.cat(candidate_pages, dim=-1)
    candidate_ids = torch.cat(candidate_ids, dim=-1)
    # Segment weights are query-independent. Compute the original BF16
    # subtraction/exp once per tile/fine/channel, rather than 256 times.
    fine_page = torch.arange(fine_z.shape[2], device=q.device) // config.coarse_group
    fine_page_z = page_z.index_select(2, fine_page)
    segment_weight = (fine_z - fine_page_z).exp().to(dtype)
    # Selection has no derivative in the original argmax path. Compute only
    # its scores without constructing a candidate-channel autograd
    # graph; re-evaluate the selected weights below with full gradient flow.
    scores = candidate_scores(route_hv, segment_weight, candidate_ids,
                              candidate_pages, visible_fine)
    selected = _topk_oldest(scores, fine_count)
    selected_id = candidate_ids.gather(-1, selected)
    selected_page = candidate_pages.gather(-1, selected)
    if route_interval != 1:
        selected_id = selected_id.repeat_interleave(route_interval, dim=1)[:, :length]
        selected_page = selected_page.repeat_interleave(route_interval, dim=1)[:, :length]
    if config.fine_writer in ("feature_mass_association", "structured_gram_association"):
        selected_z = fine_z[:, None].expand(-1, length, -1, -1, -1).gather(
            3, selected_id.unsqueeze(-1).expand(-1, -1, -1, -1, d))
        channel = feature_read_coefficients(h, selected_z)
        return coarse, selected_fine_numerator(channel, selected_id, fine_c).to(dtype)
    numerator, denominator = selected_fine_read(
        hv, segment_weight, selected_id, selected_page, fine_c)
    return coarse, (numerator / denominator).to(dtype)


def read_training(q, raw_k, v, rotated_q, rotated_k, flash,
                  route_interval=1, config=DEFAULT_CONFIG):
    """F512 local attention and per-query causal joint memory read."""
    if torch.is_grad_enabled():
        # Local attention has no dropout/RNG; RNG snapshots are unnecessary
        # and CUDA get_rng_state is not permitted during graph capture.
        local_all = checkpoint(
            lambda qr, kr, vv: _local_training_corrected_4k(
                qr, kr, vv, flash, config),
            rotated_q, rotated_k, v, use_reentrant=False, preserve_rng_state=False)
    else:
        local_all = _local_training_corrected_4k(rotated_q, rotated_k, v, flash, config)
    if q.shape[1] <= config.first_fine:
        # The first sealed fine becomes readable at first_fine.
        # Short prefixes have only exact local reads and need no writer bank.
        empty = torch.zeros_like(local_all)
        flags = [(False, tile.shape[1])
                 for tile in q.split(config.fine_tokens, dim=1)]
        return local_all, empty, empty, flags
    fine_z_all, fine_c_all, statistics = _seal_training_all(raw_k, v, config)
    summaries = _training_metadata(fine_z_all, fine_c_all, config, statistics)
    coarse_parts, fine_parts, flags = [], [], []
    group = []
    group_key = None

    def flush_group():
        if not group:
            return
        batch = q.shape[0]
        completed, _, fine_count, _ = group_key
        max_visible = max(item[4] for item in group)
        pending_width = max_visible - completed * config.coarse_group
        query = torch.cat([item[0] for item in group], dim=0)
        pz = torch.cat([item[1] for item in group], dim=0)
        pc = torch.cat([item[2] for item in group], dim=0)
        fz = torch.cat([F.pad(item[3], (0, 0, 0, max_visible - item[4]),
                              value=-torch.inf) for item in group], dim=0)
        # All tile batches share one compact bank per original batch example.
        # The indirect kernels map tile_batch % bank_batch to this bank and
        # reduce dC over tiles; no tile-bank materialization is needed.
        fc = fine_c_all.narrow(2, 0, max_visible)
        # Consecutive tiles have consecutive visible-fine counts. Device
        # construction avoids a host-to-device constant copy during capture.
        if len(group) == 1:
            # This count is identical for every batch member. Keep it as
            # static metadata rather than rematerializing a scalar CUDA
            # arange during checkpointed compiled backward.
            visible = group[0][4]
        else:
            visible = torch.arange(group[0][4], max_visible + 1, device=q.device)
            visible = visible.repeat_interleave(batch)

        def read_group(query, pz, pc, fz, fc, visible):
            return _read_training_group(query, pz, pc, fz, fc, visible,
                                        completed, pending_width, fine_count,
                                        route_interval, config)

        if torch.is_grad_enabled():
            # Softmax/routing/contraction are deterministic, with no RNG use.
            coarse, fine = checkpoint(read_group, query, pz, pc, fz, fc, visible,
                                      use_reentrant=False, preserve_rng_state=False)
        else:
            coarse, fine = read_group(query, pz, pc, fz, fc, visible)
        coarse_parts.extend(coarse.split(batch, dim=0))
        fine_parts.extend(fine.split(batch, dim=0))
        group.clear()

    # A single split backward concatenates disjoint query gradients. Repeated
    # slices otherwise allocate/zero full-sequence gradients per tile.
    for tile, query in enumerate(q.split(config.fine_tokens, dim=1)):
        start = tile * config.fine_tokens
        end = start + query.shape[1]
        visible = max(0, start // config.fine_tokens - config.visibility_tiles)
        present = bool(visible)
        if present:
            completed, pending = divmod(visible, config.coarse_group)
            _, _, fine_count = routed_counts(completed, pending, config)
            key = (completed, bool(pending), fine_count, end - start)
            if group and (key != group_key or len(group) == 8):
                flush_group()
            group_key = key
            pz, pc, fz = _tile_metadata(summaries, fine_z_all, visible, config)
            group.append((query, pz, pc, fz, visible))
        else:
            coarse_parts.append(torch.zeros_like(query))
            fine_parts.append(torch.zeros_like(query))
        flags.append((present, end - start))
    flush_group()
    return (local_all, torch.cat(coarse_parts, dim=1),
            torch.cat(fine_parts, dim=1), flags)


def _local_training_corrected_4k(q, k, v, flash, config=DEFAULT_CONFIG):
    """Causal F512 local reads; each later tile has 512 old and 256 new keys."""
    b, total, heads, d = q.shape
    if (config.fine_tokens != FINE_LENGTH or config.local_keep != 512
            or total < 3 * FINE_LENGTH or total % FINE_LENGTH):
        # Prefix scoring and native generation call full forward on short,
        # growing sequences. A partial tile sees its own prefix and at most
        # the preceding 512 exact keys; SDPA aligns the suffix queries.
        parts = []
        for start in range(0, total, config.fine_tokens):
            end = min(total, start + config.fine_tokens)
            old_start = max(0, start - config.local_keep)
            parts.append(flash.flash_attn_func(
                q[:, start:end], k[:, old_start:end], v[:, old_start:end],
                causal=True, window_size=(-1, 0)))
        return torch.cat(parts, dim=1)
    first = flash.flash_attn_func(q[:, :256], k[:, :256], v[:, :256],
                                  causal=True, window_size=(-1, 0))
    second = flash.flash_attn_func(q[:, 256:512], k[:, :512], v[:, :512],
                                   causal=True, window_size=(-1, 0))
    query = q[:, 512:].reshape(b * (total // 256 - 2), 256, heads, d)
    keys = k.unfold(1, 768, 256).permute(0, 1, 4, 2, 3).reshape(
        b * (total // 256 - 2), 768, heads, d)
    values = v.unfold(1, 768, 256).permute(0, 1, 4, 2, 3).reshape(
        b * (total // 256 - 2), 768, heads, d)
    later = flash.flash_attn_func(query, keys, values, causal=True,
                                  window_size=(-1, 0))
    return torch.cat((first, second, later.reshape(b, total - 512, heads, d)), dim=1)


class StreamCache:
    """One request's one-layer exact-local and hierarchical history state."""

    is_tao_stream = True

    def __init__(self, layer_index, route_interval=2, fusion_mode="pointwise",
                 pending_coarse_elision=False, pending_coarse_mask=False,
                 fused_fine=True, config=DEFAULT_CONFIG):
        self.config = config
        self.history_path_ablation = False
        self.history_value_ablation = False
        self.layer_index = layer_index
        self.route_interval = route_interval
        self.fusion_mode = fusion_mode
        self.fused_fine = fused_fine
        self.pending_coarse_elision = pending_coarse_elision
        self.pending_coarse_mask = pending_coarse_mask
        self.pending_only_masked_coarse_calls = 0
        self.pending_only_coarse_present_query_heads = 0
        self.pending_only_fine_present_query_heads = 0
        self.pending_coarse_elided_query_heads = 0
        self.fine_seal_events = 0
        self.fusion_pointwise_calls = 0
        self.fusion_no_history_local_returns = 0
        self.fusion_eager_calls = 0
        self.n_layers = 1
        self.layers = [LayerState(packed=True, progressive_layer=i,
                                   pending_coarse_elision=pending_coarse_elision,
                                   fused_fine=fused_fine, config=config)
                       for i in (layer_index,)]
        self.local = [None] * self.n_layers  # (normalized rotated K, V, raw K)
        self.prev_embedding = None
        self.position = 0
        self.peak_local_tokens = 0
        self._preseal_logical = [0] * self.n_layers
        self._preseal_physical = [0] * self.n_layers
        self.peak_logical_cache_bytes = 0
        self.peak_physical_cache_tensor_bytes = 0
        self.last_read_presence = None
        self.coarse_matrix_contractions = 0
        self.fine_matrix_contractions = 0
        self.fine_candidate_scores = 0
        self.route_query_heads = 0
        self.route_selection_calls = 0

    def get_pos(self):
        return self.position

    def release_execution(self, reason="request_end"):
        for layer in self.layers:
            if layer.history_region is not None:
                layer.history_region.invalidate(reason)
            if layer.page_normalization_memo is not None:
                layer.page_normalization_memo = None
                layer.page_memo_invalidations += 1

    def reset(self):
        """Discard this request's state without changing another request."""
        self.release_execution("reset")
        self.__init__(self.layer_index, self.route_interval, self.fusion_mode,
                      self.pending_coarse_elision, self.pending_coarse_mask,
                      self.fused_fine, self.config)

    def export(self):
        """Return owned tensors for a checkpoint or an independent request."""
        clone = lambda value: value.detach().clone()
        layer = self.layers[0]
        return {
            "layer_index": self.layer_index,
            "config": self.config.export(),
            "route_interval": self.route_interval,
            "fusion_mode": self.fusion_mode,
            "fused_fine": self.fused_fine,
            "position": self.position,
            "local": None if self.local[0] is None else tuple(map(clone, self.local[0])),
            "packed_fine_c": None if layer.packed_fine_c is None else clone(layer.packed_fine_c),
            "pages": [(clone(pz), None if pc is None else clone(pc), clone(fz)) for pz, pc, fz, _ in layer.pages],
            "pending_z": [clone(z) for z, _ in layer.pending],
            "pending_square_mass": None if layer.pending_square_mass is None else clone(layer.pending_square_mass),
            "pending_mass": None if layer.pending_mass is None else clone(layer.pending_mass),
            "pending_gram": None if layer.pending_gram is None else clone(layer.pending_gram),
            "pending_cross": None if layer.pending_cross is None else clone(layer.pending_cross),
            "route_ids": None if layer.route_ids is None else tuple(map(clone, layer.route_ids)),
            "route_key": layer.route_key,
            "state_version": layer.state_version,
            "route_refreshes": layer.route_refreshes,
            "fine_seal_events": self.fine_seal_events,
        }

    @classmethod
    def from_export(cls, saved):
        """Restore the same causal cursor with no alias to the exported tensors."""
        state = cls(saved["layer_index"], saved["route_interval"],
                    saved["fusion_mode"], fused_fine=saved["fused_fine"],
                    config=TaoConfig(**saved["config"]))
        clone = lambda value: value.detach().clone()
        state.position = saved["position"]
        state.local[0] = (None if saved["local"] is None else
                          tuple(map(clone, saved["local"])))
        layer = state.layers[0]
        layer.packed_fine_c = (None if saved["packed_fine_c"] is None else
                               clone(saved["packed_fine_c"]))
        layer.pages = [(clone(pz), None if pc is None else clone(pc), clone(fz),
                        layer.packed_fine_c[:, :, i * state.config.coarse_group:
                                            (i + 1) * state.config.coarse_group])
                       for i, (pz, pc, fz) in enumerate(saved["pages"])]
        offset = len(layer.pages) * state.config.coarse_group
        layer.pending = [(clone(z), layer.packed_fine_c[:, :, offset + i])
                         for i, z in enumerate(saved["pending_z"])]
        layer.pending_square_mass = (None if saved.get("pending_square_mass") is None else
                                     clone(saved["pending_square_mass"]))
        layer.pending_mass = (None if saved.get("pending_mass") is None else clone(saved["pending_mass"]))
        layer.pending_gram = (None if saved.get("pending_gram") is None else
                              clone(saved["pending_gram"]))
        layer.pending_cross = (None if saved.get("pending_cross") is None else
                               clone(saved["pending_cross"]))
        layer.route_ids = (None if saved["route_ids"] is None else
                           tuple(map(clone, saved["route_ids"])))
        layer.route_key = saved["route_key"]
        layer.state_version = saved["state_version"]
        layer.route_refreshes = saved["route_refreshes"]
        state.fine_seal_events = saved["fine_seal_events"]
        return state

    def next_chunk(self, desired):
        current = 0 if self.local[0] is None else self.local[0][0].shape[1]
        return min(desired, self.config.fine_tokens,
                   self.config.serve_limit - current)

    def process(self, layer, raw_q, raw_k, q, k, v, local_flash=None,
                allow_page_memo=False):
        if local_flash is None:
            from .flash_attention import flash_attn as local_flash

        old = self.local[layer]
        if old is None:
            all_k, all_v, all_raw = k, v, raw_k
        else:
            all_k = torch.cat((old[0], k), dim=1)
            all_v = torch.cat((old[1], v), dim=1)
            all_raw = torch.cat((old[2], raw_k), dim=1)
        if all_k.shape[1] > self.config.serve_limit:
            raise ValueError("chunk crosses a fine seal boundary")
        local = local_flash.flash_attn_func(
            q, all_k, all_v, causal=True, window_size=(-1, 0))
        state = self.layers[layer]
        coarse_pages = len(state.pages)
        _, eligible_fine, _ = routed_counts(
            len(state.pages), len(state.pending), self.config)
        query_heads = raw_q.shape[0] * raw_q.shape[1] * raw_q.shape[2]
        if not self.config.coarse_index_only:
            self.coarse_matrix_contractions += query_heads * coarse_pages
        coarse, fine, c_on, f_on = state.read(
            raw_q, self.position, self.route_interval, allow_page_memo)
        self.last_read_presence = {"start": self.position, "tokens": raw_q.shape[1],
            "completed_coarse_indexes": coarse_pages, "coarse_value_present": c_on,
            "fine_value_present": f_on}
        if state.pending and not state.pages:
            self.pending_only_fine_present_query_heads += query_heads
        self.fine_matrix_contractions += state.last_fine_contractions
        self.route_query_heads += state.last_route_queries
        self.route_selection_calls += state.last_route_queries // raw_q.shape[2]
        self.fine_candidate_scores += state.last_route_queries * eligible_fine
        self.peak_local_tokens = max(self.peak_local_tokens, all_k.shape[1])
        history_before = (self.layers[layer].logical_bytes()
                          + self.layers[layer].pending_writer_bytes())
        key_bytes = all_k.numel() * all_k.element_size()
        self._preseal_logical[layer] = history_before + 2 * key_bytes
        self._preseal_physical[layer] = history_before + 3 * key_bytes
        if all_k.shape[1] == self.config.serve_limit:
            self.fine_seal_events += 1
            fine_tokens = self.config.fine_tokens
            keep = self.config.local_keep
            self.layers[layer].seal(all_raw[:, :fine_tokens], all_v[:, :fine_tokens])
            # Own compact retained-local buffers, not views into preseal storage.
            self.local[layer] = (all_k[:, -keep:].clone(),
                                 all_v[:, -keep:].clone(),
                                 all_raw[:, -keep:].clone())
        else:
            self.local[layer] = (all_k, all_v, all_raw)
        if self.config.fine_writer in ("regularized_association", "feature_mass_association", "structured_gram_association"):
            # Include newly allocated first-pending statistics in state peaks,
            # and retain the pre-seal peak when a completed group releases them.
            history_after = state.logical_bytes() + state.pending_writer_bytes()
            local_key_bytes = self.local[layer][0].numel() * self.local[layer][0].element_size()
            self._preseal_logical[layer] = max(
                self._preseal_logical[layer], history_after + 2 * local_key_bytes)
            self._preseal_physical[layer] = max(
                self._preseal_physical[layer], history_after + 3 * local_key_bytes)
        return local, coarse, fine, c_on, f_on

    def advance(self, count):
        self.position += count
        self.peak_logical_cache_bytes = max(
            self.peak_logical_cache_bytes, sum(self._preseal_logical))
        self.peak_physical_cache_tensor_bytes = max(
            self.peak_physical_cache_tensor_bytes, sum(self._preseal_physical))

    def bytes(self):
        history = sum(layer.logical_bytes() for layer in self.layers)
        memo_tensors = [t for layer in self.layers
                        if layer.page_normalization_memo is not None
                        for t in layer.page_normalization_memo[1:] if t is not None]
        memo_bytes = sum(t.numel() * t.element_size() for t in memo_tensors)
        pending_writer = sum(layer.pending_writer_bytes() for layer in self.layers)
        pending_mass_bytes = sum(layer.pending_mass.numel() * layer.pending_mass.element_size()
                                 for layer in self.layers if layer.pending_mass is not None)
        pending_square_bytes = sum(layer.pending_square_mass.numel() * layer.pending_square_mass.element_size()
                                   for layer in self.layers if layer.pending_square_mass is not None)
        pending_cross_bytes = sum(layer.pending_cross.numel() * layer.pending_cross.element_size()
                                  for layer in self.layers if layer.pending_cross is not None)
        route_bytes = sum(t.numel() * t.element_size() for layer in self.layers
                          if layer.route_ids is not None for t in layer.route_ids)
        physical_local = sum(sum(t.numel() * t.element_size() for t in values)
                             for values in self.local if values is not None)
        if self.local[0] is None:
            local_tokens = heads = width = 0
        else:
            local_tokens = self.local[0][0].shape[1]
            heads, width = self.local[0][0].shape[2:]
        logical_local = local_tokens * self.n_layers * 2 * heads * width * 2
        unique_storages = {}

        def include(tensor):
            if tensor is None:
                return
            storage = tensor.untyped_storage()
            unique_storages[(tensor.device.type, tensor.device.index,
                             storage.data_ptr())] = storage.nbytes()

        for tensor in memo_tensors:
            include(tensor)
        for layer in self.layers:
            for tensor in (layer.pending_gram, layer.pending_cross, layer.pending_mass, layer.pending_square_mass):
                if tensor is not None:
                    include(tensor)
            for page in layer.pages:
                for tensor in page:
                    include(tensor)
            for pair in layer.pending:
                for tensor in pair:
                    include(tensor)
            if layer.packed_fine_c is not None:
                include(layer.packed_fine_c)
            if layer.route_ids is not None:
                for tensor in layer.route_ids:
                    include(tensor)
        for values in self.local:
            if values is not None:
                for tensor in values:
                    include(tensor)
        if self.prev_embedding is not None:
            include(self.prev_embedding)
        return {"history_bytes": history,
                "page_normalization_memo_bytes": memo_bytes,
                "pending_writer_bytes": pending_writer,
                "pending_gram_cross_bytes": pending_writer - pending_mass_bytes - pending_square_bytes,
                "pending_cross_bytes": pending_cross_bytes,
                "pending_square_mass_bytes": pending_square_bytes,
                "pending_feature_mass_bytes": pending_mass_bytes,
                "fused_fine": self.fused_fine,
                "fused_fine_calls": sum(layer.fused_fine_calls for layer in self.layers),
                "fusion_mode": self.fusion_mode,
                "fusion_pointwise_calls": self.fusion_pointwise_calls,
                "fusion_no_history_local_returns": self.fusion_no_history_local_returns,
                "fusion_eager_calls": self.fusion_eager_calls,
                "pending_coarse_elision": self.pending_coarse_elision,
                "pending_coarse_mask": self.pending_coarse_mask,
                "pending_only_masked_coarse_calls": self.pending_only_masked_coarse_calls,
                "pending_only_coarse_present_query_heads": (
                    self.pending_only_coarse_present_query_heads),
                "pending_only_fine_present_query_heads": (
                    self.pending_only_fine_present_query_heads),
                "pending_coarse_elided_query_heads": self.pending_coarse_elided_query_heads,
                "pending_coarse_matrix_constructions": sum(
                    layer.pending_coarse_constructions for layer in self.layers),
                "pending_coarse_elision_reads": sum(
                    layer.pending_coarse_elision_reads for layer in self.layers),
                "fine_seal_events": self.fine_seal_events,
                "route_integer_bytes": route_bytes,
                "logical_local_bf16_kv_bytes": logical_local,
                "logical_retained_bytes": history + pending_writer + logical_local,
                "physical_local_including_raw_key_bytes": physical_local,
                "allocated_cache_tensor_bytes": history + pending_writer + physical_local + route_bytes + memo_bytes,
                "unique_allocated_cache_storage_bytes": sum(unique_storages.values()),
                "local_length_tokens": local_tokens,
                "completed_coarse_pages": len(self.layers[0].pages),
                "pending_fine_states": len(self.layers[0].pending),
                "selected_fine_per_layer": [
                    0 if layer.route_ids is None else layer.route_ids[1].shape[-1]
                    for layer in self.layers],
                "peak_selected_fine_per_layer": [layer.peak_selected_fine
                                                 for layer in self.layers],
                "route_refreshes_per_layer": [layer.route_refreshes
                                              for layer in self.layers],
                "peak_local_length_tokens": self.peak_local_tokens,
                "peak_logical_cache_bytes": self.peak_logical_cache_bytes,
                "peak_physical_cache_tensor_bytes": self.peak_physical_cache_tensor_bytes,
                "coarse_matrix_contractions": self.coarse_matrix_contractions,
                "fine_matrix_contractions": self.fine_matrix_contractions,
                "fine_candidate_scores": self.fine_candidate_scores,
                "route_query_heads": self.route_query_heads,
                "route_selection_calls": self.route_selection_calls,
                "fine_matrix_repack_bytes": sum(
                    layer.fine_matrix_repack_bytes for layer in self.layers)}


