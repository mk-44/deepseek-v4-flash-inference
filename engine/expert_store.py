import math
import os
import time
from dataclasses import dataclass
import torch
import json
from arch import Checkpoint, clock, fp4_decode, Block, ModelArgs, Transformer, load_module, load_weight

ALIGN = 4096 # O_DIRECT needs file pos, length and memory address

def round_out(begin : int, end : int):
    return ALIGN * (begin // ALIGN), -(-end // ALIGN) * ALIGN

@dataclass
class ExpertDisk:
    path : str
    reads : list[tuple[int, int, int]]
    in_buffer : dict[str, tuple[int, list[int]]]
    nbytes : int

def expert_on_disk(ckpt : Checkpoint, layer : int, expert : int):
    prefix = f"layers.{layer}.ffn.experts.{expert}."
    reads, in_buffer, at = [], {}, 0
    for kind in ["scale", "weight"]:
        names = [f"{w}.{kind}" for w in ("w1", "w2", "w3")]
        entries = [ckpt.entry(prefix + n) for n in names]
        begin, end = entries[0].offset, entries[-1].offset + entries[-1].nbytes
        assert end - begin == sum(e.nbytes for e in entries), "info is not back to back in files"
        lo, hi = round_out(begin, end)
        for n, e in zip(names, entries):
            in_buffer[n] = (at + e.offset - lo, e.shape)
        reads.append((lo, hi, end - lo))
        at += (hi - lo)
    return ExpertDisk(entries[0].path, reads, in_buffer, at)

def read_expert(fd : int, where : ExpertDisk, buf : torch.Tensor):
    mem = buf.numpy()
    at, seconds = 0, []
    for begin, end, needed in where.reads:
        t = time.perf_counter()
        num_bytes_read = os.preadv(fd, [mem[at : at + end - begin]], begin)
        seconds.append(time.perf_counter() - t)
        assert num_bytes_read >= needed, f"short read: {num_bytes_read} of {needed} bytes"
        at += end - begin
    return seconds

def tensor_in(buf : torch.Tensor, where : ExpertDisk, name : str):
    at, shape = where.in_buffer[name]
    return buf[at : at + math.prod(shape)].view(shape)

def place_layout(where : ExpertDisk):
    layout, at = {}, 0
    for name, (_, shape) in where.in_buffer.items():
        layout[name] = (at, shape)
        at += math.prod(shape)
    return layout, at

def place_tensor(place : torch.Tensor, layout : dict, name : str):
    at, shape = layout[name]
    return place[at : at + math.prod(shape)].view(shape)

def copy_to_gpu(buf : torch.Tensor, where : ExpertDisk, layout : dict, place : torch.Tensor):
    for name in layout:
        place_tensor(place, layout, name).copy_(tensor_in(buf, where, name), non_blocking = True)
    done = torch.cuda.Event()
    done.record()
    return done

def decode_expert(place : torch.Tensor, layout : dict) -> dict[str, torch.Tensor]:
    return {w : fp4_decode(place_tensor(place, layout, f"{w}.weight"), place_tensor(place, layout, f"{w}.scale")) for w in ("w1", "w2", "w3")}

@dataclass
class Home:
    tier : str # "gpu", "ram", "disk"
    place : int # place in that tier, -1 for disk case.
    disk : ExpertDisk # byte info in checkpoint file.

def plan_homes(ckpt : Checkpoint, layers : list[int], n_gpu : int, n_ram : int, n_experts : int = 256):
    homes, used = {}, {"gpu" : 0, "ram" : 0}
    for l in layers:
        for e in range(n_experts):
            tier = "disk"
            if e < n_gpu:
                tier = "gpu"
            elif e < n_gpu + n_ram:
                tier = "ram"
            place = -1
            if tier != "disk":
                place = used[tier]
                used[tier] += 1
            homes[l, e] = Home(tier, place, expert_on_disk(ckpt, l, e))
    return homes

class Tiers: # allocate space in gpu and ram, done once at start.
    def __init__(self, n_gpu : int, n_ram : int, place_bytes : int):
        self.per_piece = 2 ** 30 // place_bytes # Section of 1 GiB.
        n_pieces = -(-n_ram // self.per_piece)
        self.gpu = torch.empty((n_gpu, place_bytes), dtype = torch.uint8, device = "cuda")
        self.ram = [torch.empty((self.per_piece, place_bytes), dtype = torch.uint8, pin_memory = True) for _ in range(n_pieces)]
    
    def ram_place(self, place : int):
        return self.ram[place // self.per_piece][place % self.per_piece]

def copy_to_ram(buf : torch.Tensor, where : ExpertDisk, layout : dict, place : torch.Tensor):
    for name in layout:
        place_tensor(place, layout, name).copy_(tensor_in(buf, where, name)) # CPU copy, pinned buffer -> RAM place

def fill_layer(homes : dict, tiers : Tiers, layout : dict, buf : torch.Tensor, layer : int):
    seconds = {"read" : 0.0, "gpu" : 0.0, "ram" : 0.0}
    mine = [h for (l, _), h in homes.items() if l == layer and h.tier != "disk"]
    if not mine:
        return seconds
    
    fd = os.open(mine[0].disk.path, os.O_RDONLY | os.O_DIRECT)
    for home in mine:
        seconds["read"] += sum(read_expert(fd, home.disk, buf))
        t = time.perf_counter()
        if home.tier == "gpu":
            done = copy_to_gpu(buf, home.disk, layout, tiers.gpu[home.place])
            done.synchronize() # as the next read re-uses buf.
        else:
            copy_to_ram(buf, home.disk, layout, tiers.ram_place(home.place))
        seconds[home.tier] += time.perf_counter() - t
    os.close(fd)
    return seconds

def check_places(ckpt : Checkpoint, homes : dict, tiers : Tiers, layout : dict) -> tuple[int, int]:
    good = total = 0
    for (layer, e), home in homes.items():
        if home.tier == "disk":
            continue
        place = tiers.gpu[home.place] if home.tier == "gpu" else tiers.ram_place(home.place)
        good += all(torch.equal(place_tensor(place, layout, n), ckpt.tensor(f"layers.{layer}.ffn.experts.{e}.{n}", place.device))
                    for n in layout) # read the normal way, independent of the fill
        total += 1
    return good, total

class Store:
    def __init__(self, ckpt : Checkpoint, layers : list[int], n_gpu : int, n_ram : int):
        self.homes = plan_homes(ckpt, layers, n_gpu, n_ram)
        first = self.homes[layers[0], 0].disk
        self.layout, place_bytes = place_layout(first)
        
        count = {"gpu" : 0, "ram" : 0}
        for h in self.homes.values():
            if h.tier == "disk":
                continue
            count[h.tier] += 1
        
        self.tiers = Tiers(count["gpu"], count["ram"], place_bytes)
        self.buf = torch.empty(first.nbytes, dtype = torch.uint8, pin_memory = True)
        self.working = torch.empty(place_bytes, dtype = torch.uint8, device = "cuda")
        
        for layer in layers:
            secs = fill_layer(self.homes, self.tiers, self.layout, self.buf, layer)
            print(f" filled L{layer}: read {secs['read']:.2f} s, copy to GPU {secs['gpu']:.2f} s, to RAM {secs['ram']:.2f} s |"
                  f" GPU allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        paths = sorted({h.disk.path for h in self.homes.values() if h.tier == "disk"})
        self.fds = {p : os.open(p, os.O_RDONLY | os.O_DIRECT) for p in paths} # opened once for the whole run
        self.n = {"gpu" : 0, "ram" : 0, "disk" : 0} # fetches per tier
        self.read_s = self.wait_s = 0.0 # disk tier : seconds reading and seconds waiting for the copy out of buf
    
    def stored_on_gpu(self, layer : int, expert : int):
        home = self.homes[layer, expert]
        self.n[home.tier] += 1
        if home.tier == "gpu":
            return self.tiers.gpu[home.place]
        if home.tier == "ram":
            self.working.copy_(self.tiers.ram_place(home.place), non_blocking = True)
            return self.working
        
        t = time.perf_counter()
        read_expert(self.fds[home.disk.path], home.disk, self.buf)
        t1 = time.perf_counter()
        copy_to_gpu(self.buf, home.disk, self.layout, self.working).synchronize()
        self.read_s += (t1 - t)
        self.wait_s += (time.perf_counter() - t1)
        return self.working
    
    def fetch(self, layer : int, expert : int):
        return decode_expert(self.stored_on_gpu(layer, expert), self.layout)

    def close(self):
        for fd in self.fds.values():
            os.close(fd)

    
if __name__ == "__main__":
    ckpt = Checkpoint("/v4local/ckpt")
    torch.zeros(1, device = "cuda")
    
    t = time.perf_counter()
    store = Store(ckpt, [0, 1, 2, 3], 56, 56)
    
    print(f"store ready in {time.perf_counter() - t:.1f} s | GPU allocated {torch.cuda.memory_allocated():,} B ="
      f" tier {store.tiers.gpu.numel():,} + working place {store.working.numel():,} | {len(store.fds)} files open")
    
    t = clock()
    decode_expert(store.stored_on_gpu(0, 0), store.layout) # the first decode also loads CUDA's kernels (one-time)
    print(f"first decode (one-time kernel loading): {(clock() - t) * 1e3:.0f} ms")
    
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    decoded = store.fetch(0, 200)
    print(f"one fetch from disk: keeps {(torch.cuda.memory_allocated() - base) / 2**20:.0f} MiB,"
          f" peak +{(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB")
    del decoded

    totals = {"gpu" : [], "ram" : [], "disk" : []}
    for layer, e in [(0, 0), (0, 55), (0, 56), (0, 111), (0, 112), (0, 200), (0, 255), (3, 0), (3, 56), (3, 112), (3, 255)]:
        tier = store.homes[layer, e].tier
        rows = []
        for _ in range(3):
            read0, wait0 = store.read_s, store.wait_s
            t0 = clock()
            stored = store.stored_on_gpu(layer, e)
            t1 = clock()
            decoded = decode_expert(stored, store.layout)
            t2 = clock()
            rows.append(f"{(t1 - t0) * 1e3:5.2f} + {(t2 - t1) * 1e3:4.2f}")
            totals[tier].append(t2 - t0)
        disk = f" (read {(store.read_s - read0) * 1e3:.1f} + wait {(store.wait_s - wait0) * 1e3:.2f})" if tier == "disk" else ""
        same = all(torch.equal(decoded[w], load_weight(ckpt, f"layers.{layer}.ffn.experts.{e}.{w}.weight", "cuda")) for w in decoded)
        print(f"L{layer} expert {e:>3} {tier:<4}: to the GPU + decode, ms: {' | '.join(rows)}{disk} | equal to load_weight: {same}")
        del decoded

    ms = {tier : 1e3 * sorted(v)[len(v) // 2] for tier, v in totals.items()} # the middle value per tier
    per_layer = 6 * (56 * ms["gpu"] + 56 * ms["ram"] + 144 * ms["disk"]) / 256
    print(f"per fetch (middle value): GPU {ms['gpu']:.2f} ms, RAM {ms['ram']:.2f} ms, disk {ms['disk']:.2f} ms ->"
          f" 6 experts ~{per_layer:.0f} ms per layer, ~{4 * per_layer:.0f} ms per token for 4 layers (one after another)")
    print(f"fetches {store.n} | disk: read {store.read_s:.2f} s, waited for copies {store.wait_s:.3f} s")
    store.close()