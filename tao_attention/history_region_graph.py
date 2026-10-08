"""One request/layer owned graph for the native non-refresh q1 history read.

Only execution buffers live here. Sealed semantic tensors are immutable within
state_version; append invalidates before replacing any C/z backing storage.
"""
import weakref
import torch


class HistoryRegionGraph:
    def __init__(self):
        self.graph = None
        self.stream = None
        self.raw_input = None
        self.id_input = None
        self.outputs = None
        self.epoch_key = None
        self.warm_count = 0
        self.retired_buffers = []
        self.counts = dict(epochs=0, eager_warm=0, captures=0, replays=0,
                           id_copies=0, invalidations=0)
        self.reasons = {}
        self.last_action = "uninitialized"

    def tensors(self):
        return ([t for t in (self.raw_input, self.id_input) if t is not None]
                + ([] if self.outputs is None else list(self.outputs)))

    def observation(self):
        return {**self.counts, "active_graph": self.graph is not None,
                "active_staging": self.raw_input is not None,
                "epoch_state_version": None if self.epoch_key is None else self.epoch_key[0],
                "epoch_warm_count": self.warm_count, "last_action": self.last_action,
                "invalidation_reasons": dict(self.reasons),
                "last_retired_execution_tensors_alive": sum(ref() is not None for ref in self.retired_buffers),
                "python_owned_semantic_tensor_references": 0,
                "capture_address_scope": "Current immutable semantic bank only; destroyed before seal replaces storage"}

    def invalidate(self, reason):
        if self.epoch_key is None:
            return
        # All producer and consumer work must finish before captured addresses
        # or graph-private outputs can be released. This cost stays in decode.
        torch.cuda.current_stream(self.raw_input.device).synchronize()
        self.stream.synchronize()
        self.retired_buffers = [weakref.ref(t) for t in self.tensors()]
        if self.graph is not None:
            self.graph.reset()
        self.graph = self.outputs = self.raw_input = self.id_input = None
        self.stream = self.epoch_key = None
        self.warm_count = 0
        self.counts["invalidations"] += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.last_action = "invalidate:" + reason

    def update_routes(self, selected_ids):
        if self.id_input is None:
            return
        if self.id_input.shape != selected_ids.shape:
            self.invalidate("selected_shape")
            return
        self.id_input.copy_(selected_ids)
        self.counts["id_copies"] += 1
        self.last_action = "route_copy"

    def run(self, owner, q):
        selected_ids = owner.route_ids[1]
        # z tensors are append-only immutable within state_version. No page-bank
        # scan or semantic copies are needed on an ordinary query. Imports own
        # new layer objects; refresh updates the stable ID buffer explicitly.
        bank = owner.packed_fine_c
        memo_addresses = tuple(None if value is None else
                               (value.data_ptr(), tuple(value.shape), tuple(value.stride()))
                               for value in owner.page_normalization_memo[1:])
        key = (owner.state_version, memo_addresses, tuple(selected_ids.shape), tuple(q.shape),
               q.dtype, q.device, bank.data_ptr(), tuple(bank.shape), tuple(bank.stride()))
        if key != self.epoch_key:
            self.invalidate("epoch_change")
            self.stream = torch.cuda.Stream(device=q.device)
            self.raw_input = torch.empty_like(q)
            self.id_input = torch.empty_like(selected_ids)
            self.id_input.copy_(selected_ids)
            self.counts["id_copies"] += 1
            self.epoch_key = key
            self.counts["epochs"] += 1
        current = torch.cuda.current_stream(q.device)
        if self.warm_count < 3:
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                self.raw_input.copy_(q)
                outputs = owner._q1_page_memo_region(self.raw_input, self.id_input)
            current.wait_stream(self.stream)
            for output in outputs:
                output.record_stream(current)
            self.warm_count += 1
            self.counts["eager_warm"] += 1
            self.last_action = "eager_warm"
            return outputs
        self.raw_input.copy_(q)
        if self.graph is None:
            # Capture records, rather than computes, this actual query. Device
            # quiescence is required by CUDA capture and charged to this query.
            torch.cuda.synchronize(q.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.outputs = owner._q1_page_memo_region(self.raw_input, self.id_input)
            self.counts["captures"] += 1
            self.last_action = "capture_replay"
        else:
            self.last_action = "replay"
        self.graph.replay()
        self.counts["replays"] += 1
        return self.outputs
