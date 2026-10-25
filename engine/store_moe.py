import json
import time
import torch
from torch import nn
from torch.func import functional_call
from arch import Block, ChatTokenizer, Checkpoint, Expert, MoE, ModelArgs, clock, load_module, load_weight
from expert_store import Store

class StoreExpert(nn.Module):
    def __init__(self, inner : Expert, store : Store, layer : int, expert : int):
        super().__init__()
        self.inner = inner
        self.store = store
        self.layer = layer
        self.expert = expert
    
    def forward(self, x : torch.Tensor, weights : torch.Tensor | None = None):
        # x : [n, d], weights : [n, 1] - for a particular expert
        w = self.store.fetch(self.layer, self.expert) # {"w1" : [inter, d], "w2" : [d, inter], "w3" : [inter, d]} BF16
        return functional_call(self.inner, {f"{name}.weight" : t for name, t in w.items()}, (x, weights)) # [n, d]

    
def store_block(ckpt : Checkpoint, args : ModelArgs, store : Store, layer : int, device : str = "cuda"):
    block = load_module(lambda : Block(layer, args), ckpt, f"layers.{layer}.", args, device, skip = ("ffn.experts.",))
    block.ffn.experts = nn.ModuleList([StoreExpert(e, store, layer, i) for i, e in enumerate(block.ffn.experts)])
    return block

def reference_moe(ckpt : Checkpoint, args : ModelArgs, layer : int, device = "cuda") -> MoE:
    return load_module(lambda: MoE(layer, args), ckpt, f"layers.{layer}.ffn.", args, device) # check with all 256 experts decoded at load.

class MoEWatch: # for each call of one MoE - its input and output, its time, GPU memory and what the store fetched
    def __init__(self, moe : MoE, store : Store):
        self.store = store
        moe.register_forward_pre_hook(self.before)
        moe.register_forward_hook(self.after)

    def before(self, moe, inputs):
        torch.cuda.reset_peak_memory_stats()
        self.mem0 = torch.cuda.memory_allocated()
        self.n0, self.read0, self.wait0 = dict(self.store.n), self.store.read_s, self.store.wait_s
        self.t0 = clock() # waits for the attention to finish, so only the MoE is timed

    def after(self, moe, inputs, output):
        self.seconds = clock() - self.t0
        self.kept = torch.cuda.memory_allocated() - self.mem0 # only the output should stay
        self.peak = torch.cuda.max_memory_allocated() - self.mem0
        self.x, self.ids = inputs # [b, s, d], [b, s]
        self.y = output # [b, s, d]
        self.fetched = {tier : self.store.n[tier] - self.n0[tier] for tier in self.n0}
        self.read, self.wait = self.store.read_s - self.read0, self.store.wait_s - self.wait0

def report(watch : MoEWatch, ref_seconds : float, y_ref : torch.Tensor) -> str:
    f = watch.fetched
    same = torch.equal(watch.y, y_ref)
    diff = "" if same else f" (largest difference {(watch.y.float() - y_ref.float()).abs().max().item():.3g})"
    return (f"{sum(f.values())} experts (gpu {f['gpu']}, ram {f['ram']}, disk {f['disk']}) | ours {watch.seconds * 1e3:.1f} ms"
            f" (disk read {watch.read * 1e3:.1f}, copy wait {watch.wait * 1e3:.1f}) | A's MoE {ref_seconds * 1e3:.1f} ms |"
            f" GPU kept {watch.kept:,} B, peak +{watch.peak / 2**20:.0f} MiB | equal {same}{diff}")

if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    ckpt = Checkpoint("/v4local/ckpt")
    layers = [0, 1, 2, 3]

    t = time.perf_counter()
    store = Store(ckpt, layers, 56, 56)
    print(f"store ready in {time.perf_counter() - t:.1f} s | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    t = time.perf_counter()
    embed = load_weight(ckpt, "embed.weight", "cuda") # [vocab, d] BF16
    blocks = {layer : store_block(ckpt, args, store, layer) for layer in layers}
    watches = {layer : MoEWatch(blocks[layer].ffn, store) for layer in layers}
    print(f"embedding + static weights of layers {layers} in {time.perf_counter() - t:.1f} s |"
          f" GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    chat = ChatTokenizer("deepseek_official/0731/tokenizer.json")
    texts = [[0] + chat.tokenizer.encode("The capital of France is").ids, # 0 = <｜begin▁of▁sentence｜>
             chat.encode_chat([{"role" : "user", "content" : "What is 2+2?"}])]
    prompts = [torch.tensor([tokens], device = "cuda") for tokens in texts] # each [1, s]: 6 and 11 tokens

    t = clock()
    store.fetch(0, 0) # the first decode also loads CUDA's kernels (one-time)
    print(f"first decode (one-time kernel loading): {(clock() - t) * 1e3:.0f} ms")

    with torch.inference_mode():
        hs = [embed[ids].unsqueeze(2).repeat(1, 1, args.hc_mult, 1) for ids in prompts] # each [1, s, hc, d]
        for layer in layers:
            t = time.perf_counter()
            ref = reference_moe(ckpt, args, layer)
            print(f"L{layer}: A's MoE loaded in {time.perf_counter() - t:.1f} s |"
                  f" GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            watch = watches[layer]
            for k, ids in enumerate(prompts):
                hs[k] = blocks[layer](hs[k], ids) # [1, s, hc, d]: attention, then our MoE (the watch catches its input and output)
                x = watch.x # [1, s, d]
                t = clock()
                y_ref = ref(x, ids) # [1, s, d]
                ref_seconds = clock() - t
                print(f"  prompt {k}, {ids.shape[1]} tokens, {ids.shape[1] * args.n_activated_experts} picks -> {report(watch, ref_seconds, y_ref)}")
                blocks[layer].ffn(x[:, -1:], ids[:, -1:]) # the last token alone: [1, 1, d]
                t = clock()
                y_ref = ref(x[:, -1:], ids[:, -1:]) # [1, 1, d]
                ref_seconds = clock() - t
                print(f"    last token alone, {args.n_activated_experts} picks -> {report(watch, ref_seconds, y_ref)}")
            del ref
    print(f"all fetches {store.n} (one GPU fetch was the warm-up) | disk read {store.read_s:.2f} s, copy wait {store.wait_s:.3f} s")
    store.close()
