import time
import torch
from arch import Block

NOTEBOOK_NAMES = Block.forward.__globals__ # the names Block.forward looks up while it runs. it finds hc_pre here at every call.
PLAIN_HC_PRE = NOTEBOOK_NAMES["hc_pre"] # hc_pre - about 140 small GPU jobs per call, each one launched by our thread.
RECORDINGS = {} # id of call site hc_fn weight i.e. that site's recording. each layer has two sites, before the attention and before the MoE.


class HcPreRecording: # one call site hc_pre in a 1 token step - its gpu jobs recorded once, then replayed with one call per step.
    def __init__(self, weights : tuple, consts : tuple, x : torch.Tensor, pool, stream : torch.cuda.Stream):
        self.weights, self.consts = weights, consts # (hc_fn, hc_scale, hc_base) on the gpu and (norm_eps, hc_mult, sinkhorn_iters, hc_eps).
        self.x = x.clone() # [1, 1, hc, d] the fixed input. a replay always reads from here, so each call copies its x in first.

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): # two plain calls on the stream the recording is made on, so any first call setup is done before it.
            for _ in range(2):
                PLAIN_HC_PRE(self.x, *weights, *consts)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool = pool, stream = stream): # the jobs are recorded here, not run.
            self.outputs = PLAIN_HC_PRE(self.x, *weights, *consts) # fixed outputs y [1, 1, d], post [1, 1, hc], comb [1, 1, hc, hc]. every replay writes them again
        self.graph.replay() # the first replay sets the recording up on the gpu, so no step pays for it.

    def __call__(self, x : torch.Tensor) -> tuple:
        self.x.copy_(x) # one copy job
        self.graph.replay() # one call. the GPU then runs all the recorded jobs back to back.
        return self.outputs


def recorded_hc_pre(x : torch.Tensor, hc_fn : torch.Tensor, *rest): # what Block.forward calls while the recordings are on
    r = RECORDINGS.get(id(hc_fn))
    if r is None or x.shape != r.x.shape or x.dtype != r.x.dtype: # if prompt step has more than 1 token, so it runs the plain hc_pre.
        return PLAIN_HC_PRE(x, hc_fn, *rest)
    return r(x)

def use_recordings(on : bool): # from its next call, Block.forward finds the recorded hc_pre or the plain one.
    NOTEBOOK_NAMES["hc_pre"] = recorded_hc_pre if on else PLAIN_HC_PRE

def make_recordings(blocks, args, dtype : torch.dtype): # one recording per call site, made in the order a token runs them.
    pool = torch.cuda.graph_pool_handle() # one memory pool for all of them. safe only because they replay in the order they were made.
    stream = torch.cuda.Stream()
    
    # [1, 1, hc, d] its values do not matter, a recording keeps the jobs and not the numbers
    x = torch.zeros(1, 1, args.hc_mult, args.dim, dtype = dtype, device = "cuda")
    
    for block in blocks:
        for side in ("attn", "ffn"):
            hc_fn = getattr(block, f"hc_{side}_fn")
            weights = (hc_fn, getattr(block, f"hc_{side}_scale"), getattr(block, f"hc_{side}_base"))
            consts = (block.norm_eps, block.hc_mult, block.hc_sinkhorn_iters, block.hc_eps)
            RECORDINGS[id(hc_fn)] = HcPreRecording(weights, consts, x, pool, stream)


def per_call(fn, n : int = 100) -> tuple[float, float]: # microseconds per call - our thread launching it and the GPU from the first job to the last.
    for _ in range(20): # first calls, so kernel loading lands in no measurement.
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    start.record()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    ours = (time.perf_counter() - t) / n * 1e6 # read before the wait below, so it is our thread's own work only
    end.record()
    end.synchronize()
    return ours, start.elapsed_time(end) * 1e3 / n


def same_bits(a : torch.Tensor, b : torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


if __name__ == "__main__":
    import json
    from pathlib import Path
    import expert_store
    from arch import Checkpoint, ModelArgs
    from cache_generation import CacheTrace
    from cache_store import FIXED_HOME, FOLDER, box_sizes, load_cache
    from expert_plan import load_texts, make_plan
    from full_store import saved_entries
    from generation import generate_greedy

    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    ckpt = Checkpoint(FOLDER)
    ckpt.entries.update(saved_entries(ckpt.folder))
    Path(FOLDER).mkdir(parents = True, exist_ok = True)
    box = box_sizes(ckpt, args.n_layers)
    plan = make_plan(load_texts(), box["n_gpu"], box["n_ram"], box["n_disk"])
    expert_store.COMPILED_EXPAND = False # the plain expand, so no compile: this test is about hc_pre only
    model, store, fp0 = load_cache(args, ckpt, plan, 4)
    trace = CacheTrace(dict(enumerate(model.layers)), store) # the gate hooks that move the cache, as in every run
    prompt = torch.tensor([json.load(open(FIXED_HOME))["prompts"]["sky"]["prompt"]], device = "cuda") # [1, s]

    with torch.inference_mode():
        torch.cuda.empty_cache() # so the reserved memory below counts only what the recordings hold
        reserved, t = torch.cuda.memory_reserved(), time.perf_counter()
        make_recordings(model.layers, args, model.embed.weight.dtype)
        print(f"{len(RECORDINGS)} recordings made in {time.perf_counter() - t:.2f} s "
              f"| GPU memory they hold {(torch.cuda.memory_reserved() - reserved) / 2**20:.1f} MiB")

        runs = {}
        for on in (False, True): # the same prompt twice - the plain hc_pre and then the recordings in every 1 token step.
            use_recordings(on)
            trace.calls = []
            tokens, logits, _ = generate_greedy(model, args, prompt, 8) # [1, 8], [8, vocab] - the prefill then 7 decode steps
            runs[on] = tokens, logits, trace.calls
        use_recordings(False)
        (plain_tokens, plain_logits, plain_calls), (recorded_tokens, recorded_logits, recorded_calls) = runs[False], runs[True]
        print(f"sky, 8 new tokens, plain then recorded: tokens equal {torch.equal(plain_tokens, recorded_tokens)} |"
              f" logits equal bit for bit {same_bits(plain_logits, recorded_logits)} | gate calls equal {plain_calls == recorded_calls}")

        sites = list(RECORDINGS.values()) # each fixed input still holds its site input from the last decode step.
        same = [all(same_bits(a, b) for a, b in zip(PLAIN_HC_PRE(r.x, *r.weights, *r.consts), r.outputs)) for r in sites]
        print(f"each site's last decode input, plain hc_pre against its recording, bit for bit: {same}")

        r = sites[0] # timed last - it replays one site many times and with the shared pool that can overwrite the other sites' outputs
        x = r.x.clone() # a real hidden state, the input before layer 0's attention
        for name, fn in (("plain hc_pre", lambda : PLAIN_HC_PRE(x, *r.weights, *r.consts)), ("recording", lambda : r(x))):
            ours, gpu = per_call(fn)
            print(f"{name}: our thread {ours:.1f} us per call, the GPU {gpu:.1f} us per call")
    store.close()
