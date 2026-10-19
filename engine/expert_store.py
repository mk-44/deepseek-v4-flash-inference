import math
import os
import time
from dataclasses import dataclass
import torch
from arch import Checkpoint, clock, fp4_decode, load_weight

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

def tensor_on_gpu(place : torch.Tensor, layout : dict, name : str):
    at, shape = layout[name]
    return place[at : at + math.prod(shape)].view(shape)

def copy_to_gpu(buf : torch.Tensor, where : ExpertDisk, layout : dict, place : torch.Tensor):
    for name in layout:
        tensor_on_gpu(place, layout, name).copy_(tensor_in(buf, where, name), non_blocking = True)
    done = torch.cuda.Event()
    done.record()
    return done

def decode_expert(place : torch.Tensor, layout : dict) -> dict[str, torch.Tensor]:
    return {w : fp4_decode(tensor_on_gpu(place, layout, f"{w}.weight"), tensor_on_gpu(place, layout, f"{w}.scale")) for w in ("w1", "w2", "w3")}


if __name__ == "__main__":
    ckpt = Checkpoint("/v4local/ckpt")
    torch.zeros(1, device = "cuda")
    where = expert_on_disk(ckpt, 0, 80)
    layout, place_bytes = place_layout(where) # layout same for all experts
    print(f"layout : {layout}")
    print(f"place_bytes : {place_bytes}")
    buf = torch.empty(where.nbytes, dtype = torch.uint8, pin_memory = True)
    before = torch.cuda.memory_allocated()
    place = torch.empty(place_bytes, dtype = torch.uint8, device = "cuda")
    print(f"cuda memory used : {torch.cuda.memory_allocated() - before}") 
    print(f"where : {where.nbytes}, place : {place_bytes}")
    
    fd = os.open(where.path, os.O_RDONLY | os.O_DIRECT)
    for e in (80, 88, 99):
        where = expert_on_disk(ckpt, 0, e)
        seconds = read_expert(fd, where, buf)
        t0 = time.perf_counter()
        done = copy_to_gpu(buf, where, layout, place)
        t1 = time.perf_counter()
        done.synchronize()
        t2 = time.perf_counter()
        
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        t = clock()
        decoded = decode_expert(place, layout)
        decode_ms = (clock() - t) * 1e3
        kept = torch.cuda.memory_allocated() - before
        peak = torch.cuda.max_memory_allocated() - before

        same = {w : torch.equal(decoded[w], load_weight(ckpt, f"layers.0.ffn.experts.{e}.{w}.weight", "cuda"))
                for w in decoded}
        print(f"expert {e}: read {sum(seconds) * 1e3:.1f} ms | copy: CPU back after {(t1 - t0) * 1e3:.2f} ms,"
              f" done after {(t2 - t0) * 1e3:.2f} ms ({place_bytes / (t2 - t0) / 1e9:.1f} GB/s)")
        print(f"  decode {decode_ms:.1f} ms | kept {kept:,} B, peak +{peak:,} B |"
              f" {[(w, list(x.shape), str(x.dtype)) for w, x in decoded.items()]}")
        print(f"  equal to load_weight: {same}")
        del decoded
    os.close(fd)
        