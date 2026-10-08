"""Immutable, model-independent Tao history geometry and routing schedule."""

from dataclasses import asdict, dataclass
from math import isqrt


@dataclass(frozen=True)
class TaoConfig:
    fine_tokens: int = 256
    local_keep: int = 512
    coarse_group: int = 8
    route_period: int = 24
    layer_stagger: int = 2
    budget_root_scale: int = 2
    budget_offset: int = 2
    fine_writer: str = "softmax"
    coarse_index_only: bool = False
    q1_page_normalization_memo: bool = False
    q1_history_region_graph: bool = False

    def __post_init__(self):
        if self.fine_writer not in ("softmax", "regularized_association", "feature_mass_association", "structured_gram_association"):
            raise ValueError("unsupported fine writer")
        fields = (self.fine_tokens, self.local_keep, self.coarse_group,
                  self.route_period, self.layer_stagger,
                  self.budget_root_scale, self.budget_offset)
        if not all(type(value) is int for value in fields):
            raise TypeError("Tao geometry and budget fields must be integers")
        if self.fine_tokens <= 0 or self.local_keep <= 0:
            raise ValueError("fine_tokens and local_keep must be positive")
        if self.local_keep % self.fine_tokens:
            raise ValueError("local_keep must be a multiple of fine_tokens")
        if self.coarse_group <= 0 or self.route_period <= 0:
            raise ValueError("coarse_group and route_period must be positive")
        if self.layer_stagger < 0 or self.budget_root_scale <= 0 or self.budget_offset < 0:
            raise ValueError("unsupported stagger or budget policy")

    @property
    def serve_limit(self):
        return self.local_keep + self.fine_tokens

    @property
    def first_fine(self):
        return self.serve_limit

    @property
    def first_coarse(self):
        return self.local_keep + self.coarse_group * self.fine_tokens

    @property
    def visibility_tiles(self):
        return self.local_keep // self.fine_tokens

    def budget(self, count):
        if count <= 0:
            return 0
        square = self.budget_root_scale * count
        root = isqrt(square)
        root += int(root * root < square)
        return min(count, root + self.budget_offset)

    def export(self):
        return asdict(self)
