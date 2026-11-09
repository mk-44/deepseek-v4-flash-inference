import os
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import psutil
import torch
from arch import Checkpoint, read_header
from expert_store import ExpertDisk, Tiers, copy_to_gpu, expert_on_disk, place_layout, place_tensor, read_expert
from full_store import saved_entries, static_bytes, tier_sizes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from expert_plan import E, load_texts, make_plan
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
        
        replace = dict(zip(free, newcomers)) # dropped expert -> the newcomer that takes its place
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
        self.read_s = self.wait_s = 0.0 # seconds reading the disk, seconds waiting for copies
        self.frozen = False # True during the teacher forced check
    
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
        t = time.perf_counter()
        n = os.preadv(self.out, [place.numpy()], home.slot * self.place_bytes)
        assert n == self.place_bytes, f"short read: {n} of {self.place_bytes} bytes"
        self.read_s += time.perf_counter() - t

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
    
    def check_layer(self, ckpt : Checkpoint, layer : int, experts = None) -> list[int]: # the experts whose stored bytes differ from their checkpoint file's
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


if __name__ == "__main__":
    ckpt = Checkpoint(FOLDER)
    ckpt.entries.update(saved_entries(ckpt.folder)) # where every tensor is, from the saved headers
    for l in range(4): # the downloaded files must match those headers, or every offset above is wrong
        file = ckpt.folder / ckpt.file_of[f"layers.{l}.ffn.experts.0.w1.weight"]
        assert all(ckpt.entries[n] == e for n, e in read_header(file).items()), f"{file}: header differs from the saved one"
    plan = make_plan(load_texts(), 56, 55, 191)
    torch.zeros(1, device = "cuda")
    store = CacheStore(ckpt, plan, 4) # the first 4 layers, with the last decode box's sizes
    print(f"L3's edges (first and last of tiers 1-5): {store.edges(3)}")
    t0 = time.perf_counter()
    for l in range(4):
        s = store.fill(l)
        check = f" | check L{l - 1}'s edges: differ {store.check_layer(ckpt, l - 1)}" if l > 0 else "" # one layer late
        print(f"  L{l}: read {s['read']:.1f} s, to GPU {s['gpu']:.2f} s, to RAM {s['ram']:.2f} s, to experts.bin {s['disk']:.1f} s"
              f" | experts.bin {os.path.getsize(store.path) / 1e9:.2f} GB{check}")
    print(f"  check L3's edges: differ {store.check_layer(ckpt, 3)} | 4 layers filled in {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    differ = [(l, e) for l in range(4) for e in store.check_layer(ckpt, l, range(E))]
    tiers = [h.tier for h in store.homes.values()]
    print(f"every expert, against its checkpoint bytes: " + ", ".join(f"tier {k} {tiers.count(k)}" for k in range(1, 6)) +
          f" | differ {differ} | {time.perf_counter() - t0:.0f} s")
    h = store.homes[3, plan["layers"][3]["gpu cache"][0]]
    store.place_of(h)[0] ^= 1 # on purpose: one flipped bit in a GPU place
    print(f"  one bit flipped in L3 expert {plan['layers'][3]['gpu cache'][0]}'s GPU place: differ {store.check_layer(ckpt, 3)}")
    store.place_of(h)[0] ^= 1
    h = store.homes[3, plan["layers"][3]["disk only"][0]]
    store.read_slot(h, store.stage)
    store.stage[0] ^= 1
    store.write_slot(h, store.stage) # on purpose: one flipped bit in an experts.bin slot
    print(f"  one bit flipped in L3 expert {plan['layers'][3]['disk only'][0]}'s slot: differ {store.check_layer(ckpt, 3)}")
    store.read_slot(h, store.stage) # again: check_layer used stage for every tier-5 expert it checked
    store.stage[0] ^= 1
    store.write_slot(h, store.stage)
    print(f"  both put back: differ {store.check_layer(ckpt, 3)}")
    store.close()
