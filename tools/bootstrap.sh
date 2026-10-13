#!/usr/bin/env bash
# Box setup for one Paperspace session. Run it at the start of every session; re-running is safe:
#     cd /notebooks/v4 && bash tools/bootstrap.sh
# /notebooks and /storage are network disks (Ceph) and persist. The local disk / probably doesn't (the markers
# below tell), so the Python environment lives there for fast imports and is rebuilt here when missing.
# uv, the Hugging Face cache and the logs live on /storage. The output is also saved to $V4_DATA/logs/.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$V4_DATA/bin" "$V4_DATA/logs" "$HF_HOME"
LOG="$V4_DATA/logs/bootstrap-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1
step() { printf '\n== %s  [%d s]\n' "$1" "$SECONDS"; }

step "host"
echo "$(hostname)  $(date -u +%FT%TZ)  repo commit $(git -C "$V4_ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "cpu  $(nproc) cores, $(lscpu | sed -n 's/^Model name: *//p')"
free -h | awk '/^Mem:/ {print "ram  " $2 " total, " $7 " available"}'
df -h / | awk 'NR == 2 {print "/    " $4 " free of " $2}'

step "markers (a date: survived since then; MISSING: wiped, or the first run)"
for f in /v4_marker /notebooks/.v4_marker /storage/.v4_marker; do
  printf '%-22s %s\n' "$f" "$(cat "$f" 2>/dev/null || echo MISSING)"
  date -u +%FT%TZ > "$f"
done

step "system packages"
need=()
command -v tmux >/dev/null || need+=(tmux)
command -v gcc >/dev/null || need+=(build-essential)   # Triton compiles a small C helper at run time
if [ ${#need[@]} -gt 0 ]; then
  echo "installing ${need[*]}"
  apt-get update -qq >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${need[@]}" >/dev/null
fi
echo "tmux $(tmux -V | cut -d' ' -f2), gcc $(gcc -dumpversion)"

step "uv (installs Python and packages)"
if [ ! -x "$V4_DATA/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/0.13.0/install.sh | env UV_INSTALL_DIR="$V4_DATA/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

step "Python 3.12 environment at $V4_VENV"
[ -x "$V4_VENV/bin/python" ] || uv venv -q --python 3.12 "$V4_VENV"
uv pip sync -q --python "$V4_VENV/bin/python" --torch-backend cu126 "$V4_ROOT/tools/requirements-box.txt"
python -c 'import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA available", torch.cuda.is_available())'

step "Jupyter kernel"
python -m ipykernel install --user --name v4 --display-name "v4 (Python 3.12)" >/dev/null
echo 'registered "v4 (Python 3.12)": choose it when you open a notebook on the box'
grep -qF "$V4_ROOT/tools/env.sh" ~/.bashrc 2>/dev/null || echo "source $V4_ROOT/tools/env.sh" >> ~/.bashrc

step "GPU toolchain check (DP3)"
python "$V4_ROOT/tools/gpu_env_check.py" || true

step "tiny gates on the CPU (expect 14/14 PASS, as on the Mac)"
python "$V4_ROOT/checks/gates.py" || true

step "done"
echo "log: $LOG"
