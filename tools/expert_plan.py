import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROUTES = Path("routes")
INDEX = "results/corpus_index.json"
E = 256 # num routed experts per layer


def load_texts(index : str = INDEX, folder : Path = ROUTES) -> dict[str, dict]:
    texts = {}
    
    for t in json.load(open(index))["texts"]:
        experts = torch.load(folder / f"{t['id']}.pt")["experts"].numpy() # [T, L, 6] - uint8
        answer = np.zeros(experts.shape[0], dtype = bool) # [T]
        for role, begin, end in t["spans"]:
            if role == "assistant":
                answer[begin : end] = True
        texts[t["id"]] = {"experts" : experts, "answer" : answer, "kind" : t["kind"], "shape" : t["shape"], "split" : t["split"]}
    
    return texts

def counts(experts : np.ndarray):
    L = experts.shape[1]
    flat = ((np.arange(L)[None, :, None] * E) + experts)
    return np.bincount(flat.ravel(), minlength = L * E).reshape(L, E)

def orders(texts : dict) -> dict:
    learn = [t for t in texts.values() if t["split"] == "learn"]
    use = sum(counts(t["experts"][t["answer"]]) for t in learn) # [L, E]
    tokens = sum(int(t["answer"].sum()) for t in learn)
    kinds = sorted({t["kind"] for t in learn})
    share = np.stack([sum(counts(t["experts"]) for t in learn if t["kind"] == k) for k in kinds]) # [K, L, E]
    share = share / share.sum(axis = 2, keepdims = True) # [K, L, E]
    lowest = share.min(axis = 0) # [L, E]
    return {
        "rate" : use / tokens,
        "share" : share,
        "kinds" : kinds,
        "tokens" : tokens,
        "order" : {
            "domain" : np.lexsort((-use, -lowest)), # [L, E] : highest lowest share first, tie by use
            "use" : np.argsort(-use, axis = 1, kind = "stable") # [L, E] : most used first, tie by id
        }
    }
    
def replay(texts : list[dict], order : np.ndarray, rate : np.ndarray, n_fixed : int, n_cache : int) -> tuple[float, np.ndarray]:
    reads, tokens, position_uses = 0, 0, np.zeros(n_cache)
    
    for t in texts:
        prompt, answer = t["experts"][~t["answer"]], t["experts"][t["answer"]] # [P, L, 6], [A, L, 6]
        tokens += answer.shape[0]
        prompt_counts = counts(prompt) # [L, E]
        
        for l in range(order.shape[0]):
            fixed = set(order[l, : n_fixed].tolist())
            ranked = np.lexsort((-rate[l], -prompt_counts[l])).tolist()
            cache = [e for e in ranked if e not in fixed][: n_cache]
            
            for picks in answer[:, l].tolist():
                cache_hits = sorted((e for e in picks if e in cache), key = cache.index)
                
                for e in cache_hits:
                    k = cache.index(e)
                    position_uses[k] += 1
                    cache.insert(0, cache.pop(k))
                
                for e in picks:
                    if e not in fixed and e not in cache_hits:
                        reads += 1
                        cache.insert(0, e)
                        if len(cache) > n_cache:
                            cache.pop()
                            
    return reads / tokens, position_uses / tokens


def best_split(position_uses : np.ndarray, rate : np.ndarray, fixed : np.ndarray, n_gpu : int, n_ram : int) -> int:
    n_cache = len(position_uses)
    fixed_rates = -np.sort(-np.take_along_axis(rate, fixed, axis = 1), axis = 1) # [L, F]
    
    best, most = 0, -1.0
    for n_gpu_cache in range(max(0, n_cache - n_ram), min(n_cache, n_gpu) + 1):
        on_gpu = position_uses[: n_gpu_cache].sum() + fixed_rates[:, : n_gpu - n_gpu_cache].sum()
        if on_gpu > most:
            best, most = n_gpu_cache, on_gpu
    
    return best


def make_plan(texts : dict, n_gpu : int, n_ram : int, n_disk : int):
    fast = n_gpu + n_ram
    # n_disk = (256 - n_fast) + n_cache => n_cache = n_disk - 256 + n_fast....
    
    n_cache = min(fast, n_disk - E + fast)
    assert n_cache >= 0, f"the disk holds {n_disk} places per layer, it needs at least {E - fast}"
    n_fixed = fast - n_cache
    
    stats = orders(texts)
    rate = stats["rate"]
    
    learn = [t for t in texts.values() if t["split"] == "learn" and t["answer"].any()]
    tried = {name : replay(learn, order, rate, n_fixed, n_cache) for name, order in stats["order"].items()}
    
    name = min(tried, key = lambda n : tried[n][0]) # the order with fewer disk reads on the learn answers
    
    position_uses = tried[name][1]
    fixed = stats["order"][name][:, : n_fixed] # [L, F]
    
    n_gpu_cache = best_split(position_uses, rate, fixed, n_gpu, n_ram)
    layers = []
    
    for l in range(rate.shape[0]):
        fixed_l = sorted(fixed[l].tolist(), key = lambda e : -rate[l, e]) # most used first
        others = [e for e in np.argsort(-rate[l], kind = "stable").tolist() if e not in set(fixed_l)] # most used first
        
        layers.append({
            "gpu fixed" : fixed_l[: n_gpu - n_gpu_cache], 
            "ram fixed" : fixed_l[n_gpu - n_gpu_cache :],
            "gpu cache" : others[: n_gpu_cache], 
            "ram cache" : others[n_gpu_cache : n_cache], 
            "disk only" : others[n_cache :]
        })
        
    return {
        "n_gpu" : n_gpu, 
        "n_ram" : n_ram, 
        "n_disk" : n_disk, 
        "n_cache" : n_cache, 
        "n_fixed" : n_fixed, 
        "n_gpu_cache" : n_gpu_cache,
        "order" : name, 
        "learn_reads" : {n : r for n, (r, _) in tried.items()}, 
        "position_uses" : position_uses.tolist(),
        "rate" : rate.tolist(), 
        "layers" : layers
    }

    

if __name__ == "__main__":
    n_gpu, n_ram, n_disk = (int(a) for a in sys.argv[1 : 4]) # places per layer: GPU, RAM, disk
    t0 = time.perf_counter()
    texts = load_texts()
    plan = make_plan(texts, n_gpu, n_ram, n_disk)
    n_cache, n_fixed, n_gpu_cache = plan["n_cache"], plan["n_fixed"], plan["n_gpu_cache"]
    print(f"plan in {time.perf_counter() - t0:.0f} s | {n_gpu} GPU + {n_ram} RAM + {n_disk} disk places per layer"
          f" -> {n_cache} cache places, {n_fixed} fixed experts")
    print(f"  fixed by {plan['order']} | disk reads per token on the learn answers: " + ", ".join(f"{n} {r:.2f}" for n, r in plan["learn_reads"].items()))
    print(f"  per layer: GPU {n_gpu - n_gpu_cache} fixed + {n_gpu_cache} cache | RAM {n_fixed - (n_gpu - n_gpu_cache)} fixed +"
          f" {n_cache - n_gpu_cache} cache | disk only {len(plan['layers'][0]['disk only'])}")
    stats = orders(texts)
    test = [t for t in texts.values() if t["split"] == "test" and t["answer"].any()]
    reads, _ = replay(test, stats["order"][plan["order"]], stats["rate"], n_fixed, n_cache)
    print(f"  held-out check, the 32 test answers: {reads:.2f} disk reads per token")
    for tier, experts in plan["layers"][5].items():
        print(f"  layer 5, {tier} ({len(experts)}): {experts[: 40]}")
