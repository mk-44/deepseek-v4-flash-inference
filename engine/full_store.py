import gzip
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
import psutil
import torch
from torch import nn
from arch import Checkpoint, ModelArgs, TensorEntry, Transformer, clock, load_module, read_header, target_dtype
from expert_store import Store, Tiers, copy_to_gpu, expert_on_disk, place_layout, place_tensor, plan_homes, read_expert
from store_moe import store_block
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from fetch_layers import LayerFiles, pass_prefixes

def saved_entries(folder : Path) -> dict[str, TensorEntry]:
    files = json.load(gzip.open("deepseek_official/0731/checkpoint_headers.json.gz"))["files"]
    entries = {} 
    for file, saved in files.items():
        header = json.loads(saved["header"])
        header.pop("__metadata__", None)
        
        for name, t in header.items():
            begin, end = t["data_offsets"] # counted from the end of the header
            entries[name] = TensorEntry(str(folder / file), t["dtype"], t["shape"], 8 + saved["header_bytes"] + begin, end - begin)
    return entries


def static_bytes(ckpt : Checkpoint, layers : list[int]) -> int: # GPU bytes of the always-used weights.
    ours = tuple(f"layers.{l}." for l in layers)
    
    return sum(math.prod(e.shape) * target_dtype(name, e.dtype).itemsize for name, e in ckpt.entries.items()
               if (name.startswith(ours) or not name.startswith(("layers.", "mtp."))) # our layers and the shell, not DSpark
               and ".ffn.experts." not in name and not name.endswith(".scale")) # routed experts live in the store, scales fold into their weights


def tier_sizes(
    static : int, 
    n_layers : int, 
    place_bytes : int, 
    gpu_free : int, 
    ram_free : int,
    gpu_spare : int = 2 * 2**30, 
    ram_spare : int = 8 * 2**30
) -> tuple[int, int]:
    n_gpu = (gpu_free - static - gpu_spare) // (n_layers * place_bytes) # GPU experts per layer
    pieces = (ram_free - ram_spare) // 2**30 # pinned 1 GiB pieces, keeping 8 GiB for everything else.
    n_ram = pieces * (2**30 // place_bytes) // n_layers # RAM experts per layer: 80 places per piece
    return n_gpu, n_ram

class FullStore(Store):
    def __init__(self, ckpt : Checkpoint, layers : list[int], n_gpu : int, n_ram : int, path : str):
        # Store.__init__ not called here as it fills all layers at once, from files on disk. And we download files in order while processing.
        self.homes = plan_homes(ckpt, layers, n_gpu, n_ram)
        for slot, home in enumerate(h for h in self.homes.values() if h.tier == "disk"):
            home.place = slot
        
        first_disk = self.homes[layers[0], 0].disk
        self.layout, self.place_bytes = place_layout(first_disk)
        
        self.count = {"gpu" : 0, "ram" : 0, "disk" : 0}
        for home in self.homes.values():
            self.count[home.tier] += 1
        
        self.tiers = Tiers(self.count["gpu"], self.count["ram"], self.place_bytes)
        self.buf = torch.empty(first_disk.nbytes, dtype = torch.uint8, pin_memory = True) # read from ckpt file land here.
        self.stage = torch.empty(self.place_bytes, dtype = torch.uint8, pin_memory = True) # one expert on its way, to or from repacked file.
        self.working = torch.empty(self.place_bytes, dtype = torch.uint8, device = "cuda") # every expert passes from this.
        
        self.path = path
        self.out = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT) # the repacked file.
        self.n = {"gpu" : 0, "ram" : 0, "disk" : 0} # fetches per tier.
        self.read_s = self.wait_s = 0.0 # disk tier : seconds reading and waiting for the copy out of stage.
    
    
    def fill(self, layer : int) -> dict[str, float]:
        seconds = {"read" : 0.0, "gpu" : 0.0, "ram" : 0.0, "disk" : 0.0}
        mine = [h for (l, _), h in self.homes.items() if l == layer]
        fd = os.open(mine[0].disk.path, os.O_RDONLY | os.O_DIRECT)
        
        for home in mine:
            seconds["read"] += sum(read_expert(fd, home.disk, self.buf))
            
            t = time.perf_counter()
            if home.tier == "gpu":
                copy_to_gpu(self.buf, home.disk, self.layout, self.tiers.gpu[home.place]).synchronize()
            else:
                copy_to_gpu(self.buf, home.disk, self.layout, self.working).synchronize() # seen in many tests, gpu copy to and fro total is <= ram case.
                if home.tier == "ram":
                    self.tiers.ram_place(home.place).copy_(self.working)
                else:
                    self.stage.copy_(self.working)
                    n = os.pwritev(self.out, [self.stage.numpy()], home.place * self.place_bytes) # repacked_file, stage_memory, offset
                    assert n == self.place_bytes, f"short write: {n} of {self.place_bytes} bytes"
                    
            seconds[home.tier] += time.perf_counter() - t
            
        os.close(fd)
        return seconds
    
    def stored_on_gpu(self, layer : int, expert : int) -> torch.Tensor: # [place_bytes] uint8 on the GPU : the expert bytes
        home = self.homes[layer, expert]
        if home.tier != "disk":
            return super().stored_on_gpu(layer, expert) # gpu and ram homes as in Store.
        
        self.n["disk"] += 1
        
        t = time.perf_counter()
        n = os.preadv(self.out, [self.stage.numpy()], home.place * self.place_bytes) # one aligned read. already in the place layout
        assert n == self.place_bytes, f"short read: {n} of {self.place_bytes} bytes"
        t1 = time.perf_counter()
        
        self.working.copy_(self.stage, non_blocking = True) # one copy
        done = torch.cuda.Event()
        done.record()
        done.synchronize() # the next read reuses stage
        self.read_s += t1 - t
        self.wait_s += time.perf_counter() - t1
        return self.working

    def close(self):
        os.close(self.out)
        os.remove(self.path)
            
def check_layer(ckpt : Checkpoint, store : FullStore, layer : int, experts : tuple) -> int: # how many reach the GPU with exactly the same file bytes
    good = 0
    for e in experts:
        place = store.stored_on_gpu(layer, e) # [place_bytes] on the GPU, through its tier's own path
        good += all(torch.equal(place_tensor(place, store.layout, n), ckpt.tensor(f"layers.{layer}.ffn.experts.{e}.{n}", place.device))
                    for n in store.layout) # the file read the normal way, independently of the store
    return good


def load_full(args : ModelArgs, n_layers : int, folder : str = "/v4local/ckpt", path : str = "/v4local/experts.bin"):
    Path(folder).mkdir(parents = True, exist_ok = True)
    Path(path).unlink(missing_ok = True) # repacked file left by earlier run.
    ckpt = Checkpoint(folder)
    ckpt.entries.update(saved_entries(ckpt.folder))
    layers = list(range(n_layers))
    static = static_bytes(ckpt, layers)
    _, place_bytes = place_layout(expert_on_disk(ckpt, 0, 0))
    gpu_free, ram_free = torch.cuda.mem_get_info()[0], psutil.virtual_memory().available
    n_gpu, n_ram = tier_sizes(static, n_layers, place_bytes, gpu_free, ram_free)
    n_disk = 256 - n_gpu - n_ram
    repacked = n_layers * n_disk * place_bytes
    file_bytes = max(e.offset + e.nbytes for e in ckpt.entries.values()) # the largest checkpoint file
    disk_free = shutil.disk_usage(folder).free
    
    print(f"plan, by expert id: {n_gpu} GPU + {n_ram} RAM + {n_disk} disk experts per layer, {n_layers} layers\n"
          f"  GPU {gpu_free / 1e9:.2f} GB free: static weights {static / 1e9:.2f} + tier {n_layers * n_gpu * place_bytes / 1e9:.2f} GB\n"
          f"  RAM {ram_free / 2**30:.1f} GiB available: tier {n_layers * n_ram * place_bytes / 2**30:.1f} GiB, "
          f"  pinned in {-(-n_layers * n_ram // (2**30 // place_bytes))} pieces of 1 GiB\n"
          f"  disk {disk_free / 1e9:.0f} GB free: repacked file {repacked / 1e9:.1f} GB + up to 4 checkpoint files {4 * file_bytes / 1e9:.1f} GB")
    assert disk_free > repacked + 4 * file_bytes, "not enough local disk"
    
    
    files = LayerFiles(folder, pass_prefixes(n_layers))
    t = time.perf_counter()
    store = FullStore(ckpt, layers, n_gpu, n_ram, path)
    print(f"store ready in {time.perf_counter() - t:.1f} s | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    files.wait("embed.") # l_00
    files.wait("head.") # l_45
    # l_00 and l_45 : As they are part of initial load module together so, they are downloaded initially.
    model = load_module(lambda: Transformer(args), ckpt, "", args, "cuda", skip = ("layers.", "mtp.")) # the shell
    
    # the first and last expert of each tier [gpu_s, gpu_e, ram_s, ram_e, disk_s, disk_e]
    edges = (0, n_gpu - 1, n_gpu, n_gpu + n_ram - 1, n_gpu + n_ram, 255) 
    blocks, good = [], 0
    
    for layer in layers:
        files.wait(f"layers.{layer}.") # LayerFiles keeps the last 2 files asked for: this one and the one before
        file = ckpt.folder / files.file_of[f"layers.{layer}."]
        assert all(ckpt.entries[n] == e for n, e in read_header(file).items()), f"{file}: header differs from the saved one"
        t = time.perf_counter()
        blocks.append(store_block(ckpt, args, store, layer)) # the always used weights -> in GPU
        t_static = time.perf_counter() - t
        
        s = store.fill(layer)
        check = ""
        
        if layer > 0: # one layer late, so it checks if the current layer's experts landed on the last layer's place instead.
            k = check_layer(ckpt, store, layer - 1, edges)
            good += k
            check = f" | check L{layer - 1} {k}/{len(edges)}"
        print(f"  L{layer}: static {t_static:.1f} s | experts: read {s['read']:.1f} s, to GPU {s['gpu']:.2f} s, to RAM {s['ram']:.2f} s,"
              f" to the repacked file {s['disk']:.1f} s{check} | GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB |"
              f" RAM available {psutil.virtual_memory().available / 2**30:.1f} GiB | disk free {shutil.disk_usage(folder).free / 1e9:.0f} GB")
        
    good += check_layer(ckpt, store, layers[-1], edges)
    files.close()
    print(f"checks: {good}/{n_layers * len(edges)} experts reach the GPU with exactly their file's bytes")
    assert good == n_layers * len(edges)
    model.layers = nn.ModuleList(blocks) # our layers in place of the shell's meta ones
    store.n = {"gpu" : 0, "ram" : 0, "disk" : 0} # the checks fetches don't count
    store.read_s = store.wait_s = 0.0
    return model, store



if __name__ == "__main__":
    args = ModelArgs(**json.load(open("deepseek_official/0731/inference/config.json")))
    t = time.perf_counter()
    model, store = load_full(args, args.n_layers)
    print(f"model ready in {(time.perf_counter() - t) / 60:.1f} min | GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB,"
          f" reserved {torch.cuda.memory_reserved() / 1e9:.2f}, free {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB | RAM available {psutil.virtual_memory().available / 2**30:.1f} GiB |"
          f" repacked file {os.path.getsize(store.path) / 1e9:.1f} GB, disk free {shutil.disk_usage(store.path).free / 1e9:.0f} GB")

    for tier in ("gpu", "ram", "disk"):
        keys = [key for key, h in store.homes.items() if h.tier == tier]
        picks = keys[:: max(len(keys) // 10, 1)][:10] # 10 experts spread over the layers
        read0, wait0 = store.read_s, store.wait_s
        t0 = clock()
        for layer, e in picks:
            store.stored_on_gpu(layer, e)
        ms = (clock() - t0) / len(picks) * 1e3
        split = f" (read {(store.read_s - read0) / len(picks) * 1e3:.1f} + copy {(store.wait_s - wait0) / len(picks) * 1e3:.2f})"
        print(f"  {tier:<4}: {ms:.2f} ms per expert to the GPU{split if tier == 'disk' else ''}")
    store.close()
