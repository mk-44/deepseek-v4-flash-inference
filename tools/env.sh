# Box environment. tools/bootstrap.sh sources it and adds it to ~/.bashrc, so every new terminal gets it.
# In the terminal that ran the bootstrap, run once:  source tools/env.sh
export V4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"   # the git clone, /notebooks/v4 (network disk, persists)
export V4_DATA=/storage/v4flash      # small persistent files only: uv, logs, run state (network disk, persists, billed)
export V4_LOCAL=/v4local             # local disk: model files from Hugging Face, benchmark files (free, wiped at every stop)
export V4_VENV=/root/v4venv          # Python 3.12 environment on the local disk (fast imports; rebuilt by the bootstrap if wiped)
export HF_HOME="$V4_LOCAL/hf"        # Hugging Face cache on the local disk: the model is never stored on /storage (D6)
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # no second copy of downloaded data in Hugging Face's chunk cache
export VIRTUAL_ENV="$V4_VENV"
case ":$PATH:" in
  *":$V4_VENV/bin:"*) ;;
  *) export PATH="$V4_VENV/bin:$V4_DATA/bin:$PATH" ;;
esac
