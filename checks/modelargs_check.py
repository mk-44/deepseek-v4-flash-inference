from _paths import OFFICIAL, NOTEBOOK, load_notebook_defs
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelArgs:
    # Field names match the official ModelArgs / inference/config.json.
    # Defaults are the real DeepSeek-V4-Flash values.
    vocab_size: int = 129280
    dim: int = 4096
    n_layers: int = 43
    norm_eps: float = 1e-6
    # attention (one shared KV head, K = V)
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64          # RoPE on the LAST 64 dims of each head
    q_lora_rank: int = 1024          # query latent size d_c
    o_groups: int = 8                # grouped output projection: g
    o_lora_rank: int = 1024          # d_g
    window_size: int = 128           # sliding window n_win
    compress_ratios: tuple = (0, 0) + (4, 128) * 20 + (4,)   # 0 = SWA only, 4 = CSA, 128 = HCA
    # RoPE / YaRN
    rope_theta: float = 10000.0           # pure-SWA layers, no YaRN
    compress_rope_theta: float = 160000.0 # CSA/HCA layers, with YaRN
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    max_seq_len: int = 4096          # length of the RoPE table
    # lightning indexer (CSA layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    n_hash_layers: int = 3           # layers 0..2 pick experts by token id
    moe_inter_dim: int = 2048
    score_func: str = "sqrtsoftplus" # routing score function (V4 only uses sqrt(softplus))
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # checkpoint formats: kept so the official config loads as-is; unused in the simple version
    dtype: str = "fp8"
    scale_fmt: str = "ue8m0"
    expert_dtype: str = "fp4"

    def __post_init__(self):
        self.compress_ratios = tuple(self.compress_ratios)   # JSON gives a list
        assert len(self.compress_ratios) >= self.n_layers    # official list has one extra entry for the MTP block
        assert self.rope_head_dim % 2 == 0 and self.rope_head_dim < self.head_dim
        assert self.index_head_dim & (self.index_head_dim - 1) == 0   # Hadamard needs a power of 2
        assert (self.n_heads * self.head_dim) % self.o_groups == 0
        assert self.n_shared_experts == 1
        assert self.score_func == "sqrtsoftplus"


def tiny_args() -> ModelArgs:
    return ModelArgs(
        vocab_size=512, dim=128, n_layers=6,
        n_heads=4, head_dim=64, rope_head_dim=16, q_lora_rank=64, o_groups=2, o_lora_rank=32,
        window_size=8, compress_ratios=(0, 0, 4, 16, 4, 16), max_seq_len=256,
        index_n_heads=4, index_head_dim=32, index_topk=4,
        n_routed_experts=8, n_activated_experts=2, moe_inter_dim=64,
    )


cfg = json.load(open(OFFICIAL / 'config.json'))
loaded = ModelArgs(**cfg)
default = ModelArgs()
diff = {k: (v, getattr(loaded, k)) for k, v in asdict(default).items() if v != getattr(loaded, k)}
print("loads official inference/config.json:", True)
print("fields where defaults != official config:", {k: (str(a)[:20], str(b)[:40]) for k, (a, b) in diff.items()})
print("first 43 ratios equal:", default.compress_ratios == loaded.compress_ratios[:43])
print("tiny ok:", tiny_args().n_layers, tiny_args().compress_ratios)
