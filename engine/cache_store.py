import os
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import psutil
import torch
from torch import nn
from arch import Checkpoint, read_header, clock, load_weight, ModelArgs, SequenceState, Transformer, load_module
from expert_store import ExpertDisk, Tiers, copy_to_gpu, expert_on_disk, place_layout, place_tensor, read_expert, decode_expert
from full_store import saved_entries, static_bytes, tier_sizes
from generation import RouteTrace
from store_moe import StoreExpert, store_block
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from expert_plan import E, load_texts, make_plan
from fetch_layers import LayerFiles, pass_prefixes
from collections import OrderedDict, deque

FOLDER = "/v4local/ckpt" # the checkpoint files land here during the load
DISK_SPARE = 3 * 10**9 # bytes the local disk keeps free
DISK_FILE = "/v4local/experts.bin" # the disk copies for every expert that isn't fixed.
N_FREE = 6 # GPU places with no expert - a move copies its expert up into one of these first

# Taking a complete fixed home strategy where first n_gpu, n_ram and n_disk with no cache mechanism.
FIXED_HOME = "results/full_generation_niajtmvbhu_20261025T201703Z.json" 

def box_sizes(ckpt : Checkpoint, n_layers : int, folder : str = FOLDER, gpu_spare : int = 2 * 2**30) -> dict:
    static = static_bytes(ckpt, list(range(n_layers)))
    _, place_bytes = place_layout(expert_on_disk(ckpt, 0, 0))
    gpu_free, ram_free = torch.cuda.mem_get_info()[0], psutil.virtual_memory().available
    n_gpu, n_ram = tier_sizes(static, n_layers, place_bytes, gpu_free, ram_free, gpu_spare)
    
    disk_free = shutil.disk_usage(folder).free
    n_disk = (disk_free - DISK_SPARE) // (n_layers * place_bytes)
    file_bytes = max(e.offset + e.nbytes for e in ckpt.entries.values()) # the largest checkpoint file
    peak = (E - n_gpu - n_ram) * n_layers * place_bytes + 4 * file_bytes + DISK_SPARE # during the load : disk-only experts + up to 4 downloads
    
    return {
        "n_gpu" : n_gpu, 
        "n_ram" : n_ram, 
        "n_disk" : n_disk, 
        "static" : static, 
        "place_bytes" : place_bytes,
        "gpu_free" : gpu_free, 
        "ram_free" : ram_free, 
        "disk_free" : disk_free, 
        "load_peak" : peak
    }


KEYS = ("pre t1", "pre t2", "pre t3", "pre t4", "pre t5", "pre kept", "t1", "t2", "t3", "t4", "miss", "down", "drop")

class LayerCache:
    def __init__(self, tiers : dict, rate : list[float]):
        self.gpu_fixed, self.ram_fixed = set(tiers["gpu fixed"]), set(tiers["ram fixed"])
        self.gpu = OrderedDict.fromkeys(reversed(tiers["gpu cache"])) # gpu_cache, oldest first, most recent last.
        self.ram = OrderedDict.fromkeys(reversed(tiers["ram cache"])) # ram_cache
        self.rate = rate # [E]
        assert self.gpu or not self.ram, "a RAM cache part needs a GPU part in front of it"
    
    def tier(self, e : int) -> int: # 1 GPU fixed, 2 GPU cache, 3 RAM fixed, 4 RAM cache, 5 disk only
        return 1 if e in self.gpu_fixed else 3 if e in self.ram_fixed else 2 if e in self.gpu else 4 if e in self.ram else 5
    
    def decode(self, picks : list[int], n : dict) -> list[tuple]:
        moves = []
        gpu_cache_hits = [e for e in picks if e in self.gpu]
        ram_cache_hits = [e for e in picks if e in self.ram]
        
        for e in gpu_cache_hits:
            self.gpu.move_to_end(e)
            n["t2"] += 1
        
        for e in ram_cache_hits:
            del self.ram[e]
            self.gpu[e] = None
            a, _ = self.gpu.popitem(last = False)
            self.ram[a] = None
            moves.append(("swap", e, a)) # RAM expert and gpu last swapped.
            n["t4"] += 1
            n["down"] += 1
        
        for e in picks:
            if e in self.gpu_fixed:
                n["t1"] += 1
            elif e in self.ram_fixed:
                n["t3"] += 1
            elif e not in gpu_cache_hits and e not in ram_cache_hits:
                n["miss"] += 1 # disk hit.
                
                if not self.gpu: # no gpu cache
                    continue
                
                self.gpu[e] = None
                a, _ = self.gpu.popitem(last = False) # gpu oldest
                
                if self.ram:
                    self.ram[a] = None
                    z, _ = self.ram.popitem(last = False) # RAM's oldest
                    moves.append(("miss", e, z, a)) # z dropped, e moves to a place and a moves to z.
                    n["down"] += 1
                else:
                    moves.append(("miss", e, a, None)) # no ram cache, gpu oldest is dropped
                n["drop"] += 1
        
        return moves
    
    def prefill(self, rows : list[list[int]], n : dict) -> list[tuple]:
        count = [0] * E
        
        for picks in rows:
            for e in picks:
                count[e] += 1
        
        used = sorted((e for e in range(E) if count[e] > 0), key = lambda e : (-count[e], -self.rate[e])) # most used first, ties by learn use
        
        for e in used:
            n[f"pre t{self.tier(e)}"] += 1
        
        free = [z for z in self.gpu if count[z] == 0] + [z for z in self.ram if count[z] == 0] # unused cached experts
        newcomers = [e for e in used if self.tier(e) == 5][: len(free)] # the most used experts the prompt reads from disk
        
        replace = dict(zip(free, newcomers)) # dropped expert -> the newcomer that takes its place as dict of key value pairs.
        n["pre kept"] += len(replace)
        
        for part in (self.gpu, self.ram):
            order = sorted((replace.get(e, e) for e in part), key = lambda e : count[e])
            part.clear()
            part.update((e, None) for e in order)
        return [("keep", e, z) for z, e in replace.items()]
    

@dataclass
class CacheHome:
    tier : int # 1 gpu fixed, 2 gpu cache, 3 ram fixed, 4 ram cache, 5 disk only
    place : int # gpu pool place for tiers 1 and 2, ram pool place for tiers 3 and 4, -1 for tier 5
    slot : int # place in experts.bin, -1 for tiers 1 and 3 [as fixed place], so no disk copy
    source : ExpertDisk # where its bytes are in the checkpoint file

def plan_places(ckpt : Checkpoint, plan : dict, n_layers : int) -> dict: # (layer, expert) : CacheHome
    n_gpu, n_ram, n_cache = plan["n_gpu"], plan["n_ram"], plan["n_cache"]
    n_only = E - n_gpu - n_ram # disk only experts per layer
    homes = {}
    
    for l in range(n_layers):
        t = plan["layers"][l]
        n_gpu_fixed, n_ram_fixed = len(t["gpu fixed"]), len(t["ram fixed"])
        
        for k, e in enumerate(t["gpu fixed"] + t["gpu cache"]): # layer n_gpu places
            homes[l, e] = CacheHome(1 if k < n_gpu_fixed else 2, l * n_gpu + k, -1, expert_on_disk(ckpt, l, e))
        
        for k, e in enumerate(t["ram fixed"] + t["ram cache"]): # layer n_ram places
            homes[l, e] = CacheHome(3 if k < n_ram_fixed else 4, l * n_ram + k, -1, expert_on_disk(ckpt, l, e))
        
        for k, e in enumerate(t["disk only"]): # disk only slots, during the load
            homes[l, e] = CacheHome(5, -1, l * n_only + k, expert_on_disk(ckpt, l, e))
        
        for k, e in enumerate(t["gpu cache"] + t["ram cache"]): # its cache experts slots, after every disk only one, written after the load
            homes[l, e].slot = n_layers * n_only + l * n_cache + k
        
    return homes

class CacheStore: # 5 tiers on actual bytes
    def __init__(self, ckpt : Checkpoint, plan : dict, n_layers : int, path : str = DISK_FILE):
        self.caches = [LayerCache(plan['layers'][l], plan['rate'][l]) for l in range(n_layers)]
        self.homes = plan_places(ckpt, plan, n_layers)
        first_source = self.homes[0, 0].source
        self.layout, self.place_bytes = place_layout(first_source)
        n_gpu_places = n_layers * plan["n_gpu"]
        self.tiers = Tiers(n_gpu_places + N_FREE, n_layers * plan["n_ram"], self.place_bytes) # gpu pool and ram pool.
        self.free = deque(range(n_gpu_places, n_gpu_places + N_FREE)) # gpu places that hold no expert.
        self.buf = torch.empty(first_source.nbytes, dtype = torch.uint8, pin_memory = True) # one expert read from its checkpoint file
        self.stage = torch.empty(self.place_bytes, dtype = torch.uint8, pin_memory = True) # one expert on its way to or from experts.bin
        self.working = torch.empty(self.place_bytes, dtype = torch.uint8, device = "cuda") # a ram or disk expert runs from here and then thrown away
        self.path = path
        self.out = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT) # experts.bin
        self.n = dict.fromkeys(KEYS + ("gpu", "ram", "disk"), 0) # bookkeeping counters + fetches by where the bytes are.
        self.read_s = self.wait_s = self.read_cpu_s = 0.0 # seconds reading the disk, seconds waiting for copies, cpu seconds inside those reads

        self.frozen = False # True during the teacher forced check
        self.pending = {} # ram place to cuda event, done once the copies swap queued on that place are done.
    
    def place_of(self, home : CacheHome) -> torch.Tensor: # [place_bytes] uint8 - the bytes in fast place [tiers 1 to 4]
        return self.tiers.gpu[home.place] if home.tier <= 2 else self.tiers.ram_place(home.place)
    
    def check_table(self) -> list[tuple]: # homes that break any rules
        broken, taken = [], set()
        for (l, e), h in self.homes.items():
            ok = (h.tier == self.caches[l].tier(e)) # table agrees with bookkeeping
            ok &= ((h.place >= 0) == (h.tier <= 4)) and ((h.slot >= 0) == (h.tier in (2, 4, 5))) # 1 to 4 have fast place [gpu / ram] and {2, 4, 5} have a disk copy
            for key in (("gpu" if h.tier <= 2 else "ram", h.place), ("disk", h.slot)):
                if key[1] >= 0:
                    ok &= key not in taken # no place holds two experts
                    taken.add(key)
            if not ok:
                broken.append((l, e))
        broken += [("free", k) for k in self.free if ("gpu", k) in taken or self.free.count(k) > 1] # as free place holds nobody
        return broken
    
    def close(self):
        os.close(self.out)
        os.remove(self.path)
    
    def read_slot(self, home : CacheHome, place : torch.Tensor): # a expert bytes from its experts.bin slot into a pinned place (stage or ram)
        t, c = time.perf_counter(), time.thread_time()
        n = os.preadv(self.out, [place.numpy()], home.slot * self.place_bytes)
        assert n == self.place_bytes, f"short read: {n} of {self.place_bytes} bytes"
        self.read_s += time.perf_counter() - t
        # the kernel own work for the read (pinning the pages) runs on our thread: cpu time, not waiting.
        self.read_cpu_s += time.thread_time() - c

    def write_slot(self, home : CacheHome, place : torch.Tensor): # one expert's bytes from a pinned place into its experts.bin slot
        n = os.pwritev(self.out, [place.numpy()], home.slot * self.place_bytes)
        assert n == self.place_bytes, f"short write: {n} of {self.place_bytes} bytes"

    def fill(self, layer : int) -> dict[str, float]: # the layer 256 experts from its checkpoint file to where the table says
        seconds = {"read" : 0.0, "gpu" : 0.0, "ram" : 0.0, "disk" : 0.0}
        fd = os.open(self.homes[layer, 0].source.path, os.O_RDONLY | os.O_DIRECT)
        
        for e in range(E): # in id order i.e. the order of the file
            h = self.homes[layer, e]
            seconds["read"] += sum(read_expert(fd, h.source, self.buf))
            t = time.perf_counter()
            
            if h.tier <= 2:
                copy_to_gpu(self.buf, h.source, self.layout, self.tiers.gpu[h.place]).synchronize() # tiers 1 and 2 - straight into its GPU place
            else:
                copy_to_gpu(self.buf, h.source, self.layout, self.working).synchronize() # through the GPU as its quicker than a pure CPU copy [tested earlier.]
                if h.tier <= 4:
                    self.tiers.ram_place(h.place).copy_(self.working) # tiers 3-4: into its RAM place
                else:
                    self.stage.copy_(self.working)
                    self.write_slot(h, self.stage) # tier 5: into its experts.bin slot
            seconds["gpu" if h.tier <= 2 else "ram" if h.tier <= 4 else "disk"] += time.perf_counter() - t
        os.close(fd)
        return seconds

    def edges(self, layer : int) -> list[int]: # the first and last expert of each tier's block of places
        homes = [(e, self.homes[layer, e]) for e in range(E)]
        out = []
        
        block_dict = {t : [] for t in range(1, 6)}
        
        for e, h in homes:
            t = h.tier
            if t == 5:
                block_dict[t].append((h.slot, e))
            else:
                block_dict[t].append((h.place, e))
                
        for tier in range(1, 6):
            block = sorted(block_dict[tier]) # (place, expert) in place order
            out += [block[0][1], block[-1][1]] if block else []
        return out
    
    def check_layer(self, ckpt : Checkpoint, layer : int, experts = None) -> list[int]: # experts whose stored bytes differ from their checkpoint file's
        differ = []
        for e in self.edges(layer) if experts is None else experts:
            h = self.homes[layer, e]
            if h.tier <= 4:
                place = self.place_of(h)
            else:
                self.read_slot(h, self.stage)
                place = self.stage
            if not all(torch.equal(place_tensor(place, self.layout, n), ckpt.tensor(f"layers.{layer}.ffn.experts.{e}.{n}", place.device))
                       for n in self.layout): # the file read the normal way, apart from the fill
                differ.append(e)
        
        return differ
    
    def write_copies(self) -> float: # after the load, write each cache expert's bytes from its fast place into its slot
        assert self.n["pre kept"] + self.n["down"] + self.n["drop"] == 0, "write the copies before any expert moves" 
        
        t = time.perf_counter()
        
        for h in self.homes.values():
            if h.tier == 2:
                self.stage.copy_(self.tiers.gpu[h.place]) # GPU -> pinned RAM first: the disk can't read GPU memory
                self.write_slot(h, self.stage)
            elif h.tier == 4:
                self.write_slot(h, self.tiers.ram_place(h.place)) # a RAM place is pinned and 4 KB aligned: written as it is
        return time.perf_counter() - t

    def check_copies(self) -> list[tuple]: # check if cache experts disk copy differs from their fast place
        differ = []
        for (l, e), h in self.homes.items():
            if h.tier not in (2, 4):
                continue
            self.read_slot(h, self.stage)
            if h.tier == 2:
                self.working.copy_(self.stage) # up to the GPU, to compare it with its GPU place there
                same = torch.equal(self.working, self.tiers.gpu[h.place])
            else:
                same = torch.equal(self.stage, self.tiers.ram_place(h.place))
            if not same:
                differ.append((l, e))
        return differ
    
    def stored_on_gpu(self, layer : int, expert : int) -> torch.Tensor: # [place_bytes] on gpu i.e. expert bytes from wherever it is.
        h = self.homes[layer, expert]
        
        if h.tier <= 2:
            self.n["gpu"] += 1
            return self.tiers.gpu[h.place] # already on gpu, so no copy.
        
        if h.tier <= 4:
            self.n["ram"] += 1
            self.working.copy_(self.tiers.ram_place(h.place), non_blocking = True) # one copy up, the decode after it waits on gpu, not here
            return self.working
        
        self.n["disk"] += 1
        self.read_slot(h, self.stage)
        t = time.perf_counter()
        self.working.copy_(self.stage, non_blocking = True)
        done = torch.cuda.Event()
        done.record()
        done.synchronize() # as the next disk read reuses stage
        self.wait_s += time.perf_counter() - t
        return self.working
    
    def fetch(self, layer : int, expert : int) -> dict[str, torch.Tensor]: # {"w1", "w2", "w3"}: BF16 weights on the GPU, what StoreExpert asks for
        return decode_expert(self.stored_on_gpu(layer, expert), self.layout)
    
    def wait(self, done : torch.cuda.Event): # the cpu waits until the gpu has done everything queued before 'done'
        t = time.perf_counter()
        done.synchronize()
        self.wait_s += time.perf_counter() - t
        
    def swap(self, layer : int, e : int, a : int): # e [ram cache] to free gpu place and [gpu cache] down into the ram place e left i.e. 2 copies
        he, ha = self.homes[layer, e], self.homes[layer, a]
        p, g, f = he.place, ha.place, self.free.popleft() # e's ram place, a's gpu place, gpu place that holds nobody
        
        self.tiers.gpu[f].copy_(self.tiers.ram_place(p), non_blocking = True) # e up
        self.tiers.ram_place(p).copy_(self.tiers.gpu[g], non_blocking = True) # a down, queued behind e's copy up, so it can't overwrite e's bytes.
        
        done = torch.cuda.Event()
        done.record()
        
        self.pending[p] = done # as both copies use p so the cpu must not write into p before they are done.
        self.free.append(g) # a's old gpu place holds nobody now.
        he.tier, he.place = 2, f
        ha.tier, ha.place = 4, p
    
    def take_place(self, layer : int, e : int, z : int): # e [read] from its slot takes z's fast place and z is only on its disk copy.
        he, hz = self.homes[layer, e], self.homes[layer, z]
        he.tier, he.place = hz.tier, hz.place
        hz.tier, hz.place = 5, -1
        
        if he.tier == 4: # i.e. ram place, so its pinned and 4 KB aligned, so the read lands in it directly.
            done = self.pending.pop(he.place, None)
            if done is not None:
                self.wait(done) # copies a swap queued on this place, so its must finish before the cpu overwrites it.
            self.read_slot(he, self.tiers.ram_place(he.place))
            
        else: # gpu place, so the read lands in stage and then one copy up
            self.read_slot(he, self.stage)
            self.tiers.gpu[he.place].copy_(self.stage, non_blocking = True)
            done = torch.cuda.Event()
            done.record()
            self.wait(done)
    
    def prepare(self, layer : int, rows : list[list[int]]) -> list[tuple]: 
        # one gate call picks [tokens][6], so its cache moves, done on the bytes before the MoE loop fetches.
        
        if self.frozen: # the teacher-forced check i.e. no move, no bookkeeping count
            return []
        
        cache = self.caches[layer]
        moves = cache.prefill(rows, self.n) if len(rows) > 1 else cache.decode(rows[0], self.n) # as a prompt has many tokens but a decode one step.
        
        for m in moves:
            if m[0] == "swap": # ("swap", e, a)
                self.swap(layer, m[1], m[2])
            else: # ("keep", e, z) or ("miss", e, z, a) i.e. e takes z's place (no ram part as z is the gpu oldest, a is None)
                self.take_place(layer, m[1], m[2])
                if m[0] == "miss" and m[3] is not None: # z ram place, so e sits there now and goes up like a ram hit.
                    self.swap(layer, m[1], m[3])
        return moves
    
    def reset(self): # zero every counter and timer - in place at the end of the wt load.
        for k in self.n:
            self.n[k] = 0
        self.read_s = self.wait_s = self.read_cpu_s = 0.0
    
    def restore(self, ckpt : Checkpoint, plan : dict, n_layers : int) -> int: 
        # every expert back where the load left it, bytes too. each continuous run starts from the same cache.
        torch.cuda.synchronize() # queued swaps land first
        now, self.homes = self.homes, plan_places(ckpt, plan, n_layers) # where each expert is now and where the load put it
        self.caches = [LayerCache(plan["layers"][l], plan["rate"][l]) for l in range(n_layers)]
        self.free = deque(range(n_layers * plan["n_gpu"], n_layers * plan["n_gpu"] + N_FREE))
        self.pending = {}
        moved = 0
        for key, h in self.homes.items():
            if h.tier not in (2, 4) or (now[key].tier, now[key].place) == (h.tier, h.place):
                continue # disk-only and fixed experts never move. this cache expert is still in its place
            if h.tier == 2:
                self.read_slot(h, self.stage) # its disk copy, written after the load
                self.tiers.gpu[h.place].copy_(self.stage)
            else:
                self.read_slot(h, self.tiers.ram_place(h.place)) # a ram place is pinned and 4 KB aligned, the read lands in it.
            moved += 1
        torch.cuda.synchronize()
        self.reset()
        return moved


    
    def fingerprints(self): # (layer, expert) -> number made from all its bytes but non-cryptographic.
        return {k : self.place_of(h).view(torch.int32).sum(dtype = torch.int64).item() for k, h in self.homes.items() if h.tier <= 4}
    
    
    def check_end(self, fp0 : dict) -> dict: # check at the end of the run to ensure no changes.
        torch.cuda.synchronize() # every queued copy must land before the cpu reads any ram place
        fp = self.fingerprints()
        return {
            "table" : self.check_table(), # homes that break rule
            "copies" : self.check_copies(), # cache experts whose fast place differs from their disk copy
            "changed" : [k for k in fp if k in fp0 and fp[k] != fp0[k]], # experts in a fast place now and after the load whose bytes changed
            "compared" : sum(k in fp0 for k in fp) # how many experts the fingerprints compared
        }
    

def wait_file(ckpt : Checkpoint, files : LayerFiles, prefix : str): # blocks until the file holding this prefix's tensors is on disk
    files.wait(prefix)
    file = ckpt.folder / files.file_of[prefix]
    assert all(ckpt.entries[n] == e for n, e in read_header(file).items()), f"{file} : header differs from the saved one"
    
def load_cache(args : ModelArgs, ckpt : Checkpoint, plan : dict, n_layers : int, path : str = DISK_FILE) -> tuple : # (model, store, fp0)
    t0 = time.perf_counter()
    files = LayerFiles(ckpt.folder, pass_prefixes(n_layers)) # downloads in the background, in load order, at most 2 files ahead of it.
    store = CacheStore(ckpt, plan, n_layers, path)
    split = plan['layers'][0]
    
    print(f"{plan['n_gpu']} GPU + {plan['n_ram']} RAM + {plan['n_disk']} disk places: C = {plan['n_cache']} | per layer GPU {len(split['gpu fixed'])} fixed"
          f" + {len(split['gpu cache'])} cache, RAM {len(split['ram fixed'])} fixed + {len(split['ram cache'])} cache, {len(split['disk only'])} disk only")
    
    print(f"store ready in {time.perf_counter() - t0:.1f} s | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    wait_file(ckpt, files, "embed.")
    wait_file(ckpt, files, "head.")
    model = load_module(lambda : Transformer(args), ckpt, "", args, "cuda", skip = ("layers.", "mtp.")) # embedding, final_norm, hc_head, head
    
    
    blocks, differ = [], []
    for l in range(n_layers):
        wait_file(ckpt, files, f"layers.{l}.") # LayerFiles keeps the last 2 files asked for i.e. current layer and the one before.
        t = time.perf_counter()
        blocks.append(store_block(ckpt, args, store, l))
        t_static = time.perf_counter() - t
        s = store.fill(l)
        
        check = ""
        
        if l > 0: # check to ensure that this layer fill must not have landed in last layer's place by mistake.
            bad = store.check_layer(ckpt, l - 1)
            differ += [(l - 1, e) for e in bad]
            check = f" | check L{l - 1}: differ {bad}"
            
        print(f"  L{l}: static {t_static:.1f} s | experts: read {s['read']:.1f} s, to GPU {s['gpu']:.2f} s, to RAM {s['ram']:.2f} s,"
              f" to experts.bin {s['disk']:.1f} s{check} | GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB |"
              f" RAM available {psutil.virtual_memory().available / 2**30:.1f} GiB | disk free {shutil.disk_usage(path).free / 1e9:.0f} GB")
    
    differ += [(n_layers - 1, e) for e in store.check_layer(ckpt, n_layers - 1)] # for the last layer, while its file is still on disk.
    files.close() # deletes the last downloads and its room to be used by the cache copies.
    
    n_edges = sum(len(store.edges(l)) for l in range(n_layers))
    print(f"edge checks: {n_edges - len(differ)}/{n_edges} experts hold exactly their checkpoint file's bytes")
    assert not differ, f"experts whose bytes differ: {differ}"

    
    seconds = store.write_copies() # write cache experts bytes into its experts.bin slot, before we make any moves.
    t = time.perf_counter()
    copies = store.check_copies() # cache expert disk copy read back and compared with its fast place.
    print(f"cache copies: {sum(h.tier in (2, 4) for h in store.homes.values()):,} written in {seconds:.1f} s,"
          f" read back in {time.perf_counter() - t:.1f} s | differ {copies}")
    assert not copies, "a cache expert's disk copy differs from its fast place"
    
    for e in store.edges(0)[::2]: # one fetch from the first place of each tier, to warm up.
        store.fetch(0, e)
    
    t = time.perf_counter()
    fp0 = store.fingerprints() # every fast expert bytes as a number.
    print(f"fingerprints of {len(fp0):,} experts in fast places in {time.perf_counter() - t:.1f} s")
    store.reset() # load's reads and fetches don not count in the run.
    
    model.layers = nn.ModuleList(blocks) # our layers replace the shell's meta ones.
    print(f"load done in {(time.perf_counter() - t0) / 60:.1f} min | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB,"
          f" free {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB | RAM available {psutil.virtual_memory().available / 2**30:.1f} GiB |"
          f" experts.bin {os.path.getsize(path) / 1e9:.1f} GB, disk free {shutil.disk_usage(path).free / 1e9:.0f} GB")
    return model, store, fp0



if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    ckpt = Checkpoint(FOLDER)
    ckpt.entries.update(saved_entries(ckpt.folder)) # where every tensor is, from the saved headers: the files themselves come during the load
    plan = make_plan(load_texts(), 56, 55, 191) # the same plan as parts 7 and 8, so every count below is known
    model, store, fp0 = load_cache(args, ckpt, plan, 4)
    
    wired = all(isinstance(x, StoreExpert) and x.store is store and (x.layer, x.expert) == (l, e)
                for l, block in enumerate(model.layers) for e, x in enumerate(block.ffn.experts))
    
    meta = [n for n, w in [*model.named_parameters(), *model.named_buffers()]
            if w.is_meta and ".inner." not in n and not n.startswith("mtp.")]
    
    print(f"model: {len(model.layers)} layers | every expert asks the store for its own (layer, expert): {wired} | weights still on meta: {meta}"
          f" | checkpoint files left on disk: {sorted(p.name for p in Path(FOLDER).glob('*.safetensors'))}")
    
    
    trace = RouteTrace(dict(enumerate(model.layers))) # records each gate call as {layer, experts, weights}, as the fixed_home run did
    t = clock()
    
    trace = RouteTrace(dict(enumerate(model.layers))) # records each gate call as {layer, experts, weights}, as the fixed_home run did
    t = clock()
    for name, run in json.load(open(FIXED_HOME))["prompts"].items(): # the fixed_home run: each prompt's ids and gate calls
        trace.calls = []
        state = SequenceState(args, 1, model.embed.weight.dtype, device = "cuda")
        model(torch.tensor([run["prompt"]], device = "cuda"), state) # the prompt's prefill through our 4 layers, as generate_greedy's first step does it
        theirs = run["calls"][:4] # the run's prefill went through layers 0-42 first: its calls 0-3 are layers 0-3
        same = [ours == call for ours, call in zip(trace.calls, theirs)]
        diff = max(abs(a - b) for ours, call in zip(trace.calls, theirs) for x, y in zip(ours["weights"], call["weights"]) for a, b in zip(x, y))
        print(f"  {name}, {len(run['prompt'])} tokens: layers 0-3 gate calls equal to the fixed_home run's {same} | largest weight difference {diff:.3g}")
    print(f"prefill of 4 prompts through 4 layers: {clock() - t:.1f} s | fetches {({k : store.n[k] for k in ('gpu', 'ram', 'disk')})}"
          f" | read {store.read_s:.2f} s, waited {store.wait_s:.2f} s")
    print(f"end check: {store.check_end(fp0)}")
    store.close()
    
