"""Phase 5, iteration A: the checkpoint's files on the local disk just in time, for a layer-by-layer pass.
    files = LayerFiles("/v4local/ckpt")       # starts downloading in the background
    waited = files.wait("layers.3.")          # blocks until the file with layers.3.* is on disk; prints its download
The files come from Hugging Face (the pinned revision, anonymous) in the order a pass needs them: the embedding, the
head (with the final norm and hc_head), then layers 0-42. The DSpark files are not needed. The local disk can't hold all
156 GB, so this is a rolling window:
- the download runs at most `ahead` files past the last file asked for;
- a file is deleted once `keep` newer files have been asked for: by then the pass, and DeepSeek's reference reading
  the same file, are done with it.
A file already in the folder (from an earlier run in this session) is used as it is."""
import json
import os
import shutil
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO, REVISION = 'deepseek-ai/DeepSeek-V4-Flash-0731', '7872f01b1d1fe23eabc4c98b48bffcef5a386062'


def pass_prefixes(n_layers = 43):
    return ['embed.', 'head.'] + [f'layers.{i}.' for i in range(n_layers)]


class LayerFiles:
    def __init__(self, folder, prefixes = None, ahead = 2, keep = 2, download = None):
        """prefixes: the files to fetch, in order, named by a tensor-name start (default: a whole pass).
        download(name, folder): fetches one file (default: from Hugging Face); the checks pass a stand-in."""
        self.folder = Path(folder).expanduser()
        self.folder.mkdir(parents = True, exist_ok = True)
        file_of = json.load(open(ROOT / 'deepseek_official' / '0731' / 'model.safetensors.index.json'))['weight_map']
        self.prefixes = prefixes or pass_prefixes()
        self.file_of = {p : next(f for n, f in file_of.items() if n.startswith(p)) for p in self.prefixes}
        self.order = [self.file_of[p] for p in self.prefixes]
        self.ahead, self.keep = ahead, keep
        self.download = download or hf_download
        self.ready = {}          # file -> (bytes, seconds to download; 0 if it was already there)
        self.asked = []          # files in the order they were asked for
        self.deleted = set()
        self.error = None
        self.cond = threading.Condition()
        print(f'{len(self.order)} files to {self.folder}, {shutil.disk_usage(self.folder).free / 1e9:.0f} GB free there')
        self.thread = threading.Thread(target = self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for j, name in enumerate(self.order):
                with self.cond:
                    self.cond.wait_for(lambda: j < len(self.asked) + self.ahead)
                path = self.folder / name
                seconds = 0.0
                if not path.exists():
                    t = time.perf_counter()
                    self.download(name, self.folder)
                    seconds = time.perf_counter() - t
                with self.cond:
                    self.ready[name] = (path.stat().st_size, seconds)
                    self.cond.notify_all()
        except Exception as e:
            with self.cond:
                self.error = e
                self.cond.notify_all()

    def wait(self, prefix):
        """Blocks until the file holding the tensors that start with `prefix` is on disk; returns the seconds waited."""
        name = self.file_of[prefix]
        if name in self.deleted:
            raise RuntimeError(f'{name} was already used and deleted: make a new LayerFiles to fetch it again')
        t = time.perf_counter()
        with self.cond:
            if name not in self.asked:
                self.asked.append(name)
                self.cond.notify_all()                              # the download may now run further ahead
            self.cond.wait_for(lambda: name in self.ready or self.error is not None)
            if self.error is not None and name not in self.ready:
                raise RuntimeError(f'downloading {name} failed: {self.error!r}')
            old = [f for f in self.asked[:-self.keep] if f not in self.deleted]
        waited = time.perf_counter() - t
        for f in old:
            (self.folder / f).unlink(missing_ok=True)
            self.deleted.add(f)
        nbytes, seconds = self.ready[name]
        how = f'downloaded in {seconds:.1f} s ({nbytes / seconds / 1e9:.2f} GB/s)' if seconds else 'already on disk'
        print(f'  file {name} ({prefix}*): {nbytes / 1e9:.2f} GB {how}, waited {waited:.1f} s')
        return waited

    def close(self):
        """Deletes the files still on disk."""
        for f in self.order:
            if f not in self.deleted:
                (self.folder / f).unlink(missing_ok=True)
                self.deleted.add(f)


def hf_download(name, folder):
    """One file of the pinned checkpoint from Hugging Face, anonymously (no token is read or sent). The cache stays
    on the local disk: the model is never stored on /storage (D6)."""
    if os.environ.get('HF_HOME', '').startswith('/storage'):
        os.environ['HF_HOME'] = str(Path(folder).parent / 'hf')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
    from huggingface_hub import hf_hub_download
    hf_hub_download(REPO, name, revision=REVISION, local_dir=folder, token=False)
