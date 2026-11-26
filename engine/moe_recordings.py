import time
import types
import itertools
import torch
from torch.func import functional_call
from arch import MoE
from expert_store import decode_expert

MOE_RECORDINGS = {}

class MoERecording: # a layer MoE in a token step, from the gate output onwards, recorded once, then replayed with one call per step.
    on = False 
    # True - token steps replay the recordings. False - every MoE call runs the plain MoE.forward.
    orders = None # [720, 6] on the gpu, every order of the gate's 6 positions. made once in make_moe_recordings
    order_rows = {} # an order (a tuple) to its row in orders, so finding it is one dict lookup

    def __init__(self, moe : MoE, store, layer : int, slots : torch.Tensor, x : torch.Tensor, w : torch.Tensor, pool, stream : torch.cuda.Stream):
        self.moe, self.store, self.layer = moe, store, layer
        self.slots, self.x, self.w = slots, x, w # shared by all layers: [6, place_bytes] the 6 experts bytes, [1, d] the input, [6] their routing weights
        self.expert = moe.experts[0].inner # every routed expert has the same shapes and swiglu limit, so one module runs all 6, each with its own weights

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): # two plain runs on the stream the recording is made on, so any first call setup is done before it.
            for _ in range(2):
                self.after_gate()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool = pool, stream = stream): # the jobs are recorded here, not run.
            self.y = self.after_gate() # [1, d] the fixed output. every replay writes it again
        self.graph.replay() # the first replay sets the recording up on the gpu, so no step pays for it.
    
    
    def after_gate(self) -> torch.Tensor: # [1, d] what MoE.forward does after its gate for 1 token, with its 6 experts in the slots
        y = torch.zeros_like(self.x, dtype = torch.float32) # [1, d] as MoE.forward makes it
        for k in range(len(self.slots)): # slot order is ascending expert number, the order MoE.forward adds them in
            w = decode_expert(self.slots[k], self.store.layout) # {"w1", "w2", "w3"} BF16, the compiled expand
            y += functional_call(self.expert, {f"{n}.weight" : t for n, t in w.items()}, (self.x, self.w[k : k + 1, None])) # [1, d]
            del w # this expand is freed before the next one is made, so the pool holds one expand, not two
        y += self.moe.shared_experts(self.x) # added last, as in MoE.forward
        return y.type_as(self.x)
    
    def __call__(self, weights : torch.Tensor, picks : list[int], x : torch.Tensor) -> torch.Tensor: # weights [1, 6], picks the 6 experts in the gate's order, x [1, d]
        order = sorted(range(len(picks)), key = lambda k : picks[k]) # positions in the gate output, by ascending expert number
        for slot, k in enumerate(order):
            self.store.copy_into(self.layer, picks[k], self.slots[slot]) # one copy job per expert
        torch.index_select(weights[0], 0, self.orders[self.order_rows[tuple(order)]], out = self.w) # [6] the routing weights in slot order. one gpu job, our thread does not wait
        self.x.copy_(x)
        self.graph.replay() # one call. the gpu then runs the 6 expands, the 6 experts, the shared expert and the sum back to back
        return self.y


    
def recorded_moe(moe : MoE, x : torch.Tensor, input_ids : torch.Tensor) -> torch.Tensor: # MoE forward from now on. x : [b, s, d], input_ids : [b, s]
    r = MOE_RECORDINGS.get(id(moe))
    if not MoERecording.on or r is None or x.shape[0] * x.shape[1] != 1: # a prompt step or the check has more than 1 token
        return MoE.forward(moe, x, input_ids)
    
    weights, indices = moe.gate(x.view(-1, moe.dim), input_ids.flatten()) # [1, 6], [1, 6]. its hook moves the cache and hands the picks to the store
    layer, rows = r.store.picks
    
    if layer != r.layer: # no gate hook ran for this call, so read the picks here
        rows = indices.tolist()
    return r(weights, rows[0], x.view(-1, moe.dim)).view(x.shape)


def make_moe_recordings(blocks, store, dtype : torch.dtype): # one recording per layer, made in the order every token replays them
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream() # one memory pool for all of them. safe only because they replay in the order they were made.
    moe = blocks[0].ffn
    
    slots = torch.zeros(moe.gate.topk, store.place_bytes, dtype = torch.uint8, device = "cuda") # [6, place_bytes]. zero bytes expand to zero weights, so the first runs make no inf or nan
    x = torch.zeros(1, moe.dim, dtype = dtype, device = "cuda") # [1, d]
    w = torch.zeros(moe.gate.topk, dtype = torch.float32, device = "cuda") # [6]
    every = list(itertools.permutations(range(moe.gate.topk))) # all 720 orders of 6 positions
    MoERecording.orders = torch.tensor(every, device = "cuda") # [720, 6] sent up once here, so no step sends an index list up
    MoERecording.order_rows = {o : row for row, o in enumerate(every)}
    
    for layer, block in enumerate(blocks):
        MOE_RECORDINGS[id(block.ffn)] = MoERecording(block.ffn, store, layer, slots, x, w, pool, stream)
        block.ffn.forward = types.MethodType(recorded_moe, block.ffn) # it runs the plain MoE.forward whenever MoERecording.on is False

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
    from recordings import per_call, same_bits
    from store_moe import MoEWatch
    
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    ckpt = Checkpoint(FOLDER)
    ckpt.entries.update(saved_entries(ckpt.folder))
    Path(FOLDER).mkdir(parents = True, exist_ok = True)
    box = box_sizes(ckpt, args.n_layers)
    plan = make_plan(load_texts(), box["n_gpu"], box["n_ram"], box["n_disk"])
    expert_store.COMPILED_EXPAND = False # the load runs the plain expand, as arms.py does
    model, store, fp0 = load_cache(args, ckpt, plan, 4)
    trace = CacheTrace(dict(enumerate(model.layers)), store) # the gate hooks that move the cache and hand the picks to the store, as in every run
    MoEWatch.memory = False
    watches = [MoEWatch(block.ffn, store) for block in model.layers] # each keeps its layer's last MoE input and output
    prompt = torch.tensor([json.load(open(FIXED_HOME))["prompts"]["sky"]["prompt"]], device = "cuda") # [1, s]
    
    

    with torch.inference_mode():
        expert_store.COMPILED_EXPAND = True # the recordings hold the compiled expand
        store.fetch(0, next(e for e in range(256) if store.homes[0, e].tier <= 2)) # compiles both matrix shapes once, as arms.py does
        torch.cuda.empty_cache() # so the reserved memory below counts only what the recordings hold
        reserved, t = torch.cuda.memory_reserved(), time.perf_counter()
        
        make_moe_recordings(model.layers, store, model.embed.weight.dtype)
        
        print(f"{len(MOE_RECORDINGS)} MoE recordings made in {time.perf_counter() - t:.2f} s "
              f"| GPU memory they hold {(torch.cuda.memory_reserved() - reserved) / 2**20:.0f} MiB")
        
        
        runs = {}
        for on in (False, True): # the same prompt twice: the plain MoE, then the recordings in every 1 token step
            MoERecording.on = on
            trace.calls = []
            tokens, logits, _ = generate_greedy(model, args, prompt, 8) # [1, 8], [8, vocab]: the prefill then 7 decode steps
            runs[on] = tokens, logits, trace.calls
        (plain_tokens, plain_logits, plain_calls), (recorded_tokens, recorded_logits, recorded_calls) = runs[False], runs[True]
        print(f"sky, 8 new tokens, plain then recorded: tokens equal {torch.equal(plain_tokens, recorded_tokens)} |"
              f" logits equal bit for bit {same_bits(plain_logits, recorded_logits)} | gate calls equal {plain_calls == recorded_calls}")
        
        
        


        same = [same_bits(MoE.forward(block.ffn, watch.x, watch.ids), watch.y) for block, watch in zip(model.layers, watches)] 
        # each layer's last decode step, still in its recording's fixed output
        
        print(f"each layer's last decode input, the plain MoE against its recording, bit for bit: {same}")

        moe, watch = model.layers[0].ffn, watches[0] # timed last: replaying one layer many times can overwrite the other layers' outputs in the shared pool
        x, ids = watch.x.clone(), watch.ids.clone() # a real decode input of layer 0's MoE
        weights, indices = moe.gate(x.view(-1, moe.dim), ids.flatten()) # its routing, to time the recording alone
        r, picks = MOE_RECORDINGS[id(moe)], indices[0].tolist()
        
        for name, on, fn in (("plain MoE", False, lambda : moe(x, ids)), ("recorded MoE", True, lambda : moe(x, ids)),
                             ("the recording alone", True, lambda : r(weights, picks, x.view(-1, moe.dim)))):
            MoERecording.on = on
            ours, gpu = per_call(fn, 30)
            print(f"{name}: our thread {ours:.0f} us per call, the GPU {gpu:.0f} us per call")
        MoERecording.on = False
    
    store.close()