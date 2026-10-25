import json
import os
import time
import torch
from torch import nn
from arch import ChatTokenizer, Checkpoint, ModelArgs, SequenceState, Transformer, clock, load_module
from expert_store import Store
from store_moe import MoEWatch, store_block

class RouteTrace:
    def __init__(self, blocks : dict):
        self.calls = []
        for layer, block in blocks.items():
            block.ffn.gate.register_forward_hook(self.recorder(layer))
    
    def recorder(self, layer : int):
        def record(gate, inputs, outputs):
            weights, indices = outputs # [n, 6], [n, 6]
            self.calls.append({"layer" : layer, "experts" : indices.tolist(), "weights" : weights.tolist()})
        return record

def generate_greedy(model : Transformer, args : ModelArgs, ids : torch.Tensor, n_new : int, after_step = None, stop : int | None = None):
    state = SequenceState(args, 1, model.embed.weight.dtype, device = ids.device)
    tokens, scores, seconds, x = [], [], [], ids
    for step in range(n_new):
        t = clock()
        logits = model(x, state) # [1, vocab]
        seconds.append(clock() - t)
        x = logits.argmax(dim = -1, keepdim = True) # [1, 1]
        tokens.append(x)
        scores.append(logits)
        
        if after_step:
            after_step(step, seconds[-1], x)
        
        if stop is not None and x.item() == stop:
            break
        
    return torch.cat(tokens, dim = -1), torch.cat(scores), seconds # [1, n_new], [n_new, vocab], n_new

def teacher_forced_check(model : Transformer, ids : torch.Tensor, tokens : torch.Tensor, scores : torch.Tensor):
    # does prefill over the prompt + the generated tokens must predict the same tokens the generation picked
    full_ids = torch.cat([ids, tokens[:, : -1]], dim = 1) # [1, s + n_new - 1]
    logits = model(full_ids, return_all_logits = True)[0, ids.shape[1] - 1 :] # [n_new, vocab]
    same = logits.argmax(dim = -1) == tokens[0] # [n_new]
    diff = (logits - scores).abs().max(dim = -1).values # [n_new]
    top2 = scores.topk(2, dim = -1).values # [n_new, 2]
    return same, diff, top2[:, 0] - top2[:, 1] # [n_new] each


if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    ckpt = Checkpoint("/v4local/ckpt")
    layers = [0, 1, 2, 3]
    
    t = time.perf_counter()
    store = Store(ckpt, layers, 56, 56)
    model = load_module(lambda: Transformer(args), ckpt, "", args, "cuda", skip = ("layers.", "mtp.")) # embedding, norm, hc_head, head
    blocks = {layer : store_block(ckpt, args, store, layer) for layer in layers}
    model.layers = nn.ModuleList(blocks.values()) # our 4 layers in place of the shell's 43 meta ones
    watches = {layer : MoEWatch(block.ffn, store) for layer, block in blocks.items()}
    trace = RouteTrace(blocks)
    print(f"model ready in {time.perf_counter() - t:.1f} s | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    chat = ChatTokenizer("deepseek_official/0731/tokenizer.json")
    texts = [
        [0] + chat.tokenizer.encode("The capital of France is").ids, # 0 = <|begin_of_sentence|>
        chat.encode_chat([{"role" : "user", "content" : "What is 2+2?"}])
    ]
    prompts = [torch.tensor([tokens], device = "cuda") for tokens in texts] # each [1, s]: 6 and 11 tokens
    store.fetch(0, 0) # the first decode also loads CUDA's kernels (one-time)
    
    def show(step, seconds, token): # after each step: its time, each layer's MoE time and fetches (gpu/ram/disk), the token
        moe = " | ".join(f"L{l} {w.seconds * 1e3:.0f} ms ({w.fetched['gpu']}/{w.fetched['ram']}/{w.fetched['disk']})"
                         for l, w in watches.items())
        print(f"  step {step}: {seconds * 1e3:.0f} ms | MoE {moe} | -> {chat.tokenizer.decode(token[0].tolist())!r}")
    
    routes = []
    with torch.inference_mode():
        for k, ids in enumerate(prompts):
            print(f"prompt {k}, {ids.shape[1]} tokens:")
            trace.calls = []
            tokens, scores, seconds = generate_greedy(model, args, ids, 10, show)
            routes.append({"prompt" : texts[k], "new_tokens" : tokens[0].tolist(), "calls" : trace.calls})
            trace.calls = [] # the check below is not part of the trace
            print(f"  prefill {seconds[0]:.2f} s, then {sum(seconds[1:]) / len(seconds[1:]) * 1e3:.0f} ms per token |"
                  f" text {chat.tokenizer.decode(tokens[0].tolist())!r}")
            same, diff, margin = teacher_forced_check(model, ids, tokens, scores)
            print(f"  check, one prefill over prompt + new tokens: same tokens {int(same.sum())}/{len(same)} |"
                  f" largest score difference {diff.max():.3g} | smallest top-2 margin {margin.min():.3g}")
    path = f"results/routes_{os.uname().nodename}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    json.dump({"layers" : layers, "prompts" : routes}, open(path, "w"))
    print(f"routing trace: {sum(len(r['calls']) for r in routes)} MoE calls -> {path}")
    store.close()