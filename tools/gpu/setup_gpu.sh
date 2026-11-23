#!/bin/bash
# Installation du serveur d'entrainement NNUE (RTX 5090, Blackwell sm_120).
# Idempotent : peut etre relance. Journal : ~/setup_gpu.log
set -e
W=${W:-/workspace}
mkdir -p "$W" && cd "$W"
log() { echo "$(date +%H:%M:%S) $*" | tee -a ~/setup_gpu.status; }

log "paquets systeme"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs zstd curl build-essential g++-14 cmake python3-venv python3-dev tmux >/dev/null
git lfs install --skip-repo >/dev/null

log "telechargement du binpack, decompresse a la volee (disque de 32 Go)"
mkdir -p "$W/data"
HF=https://huggingface.co/datasets/linrock/test80-2024/resolve/main
BINPACKS=${BINPACKS:-"test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst"}
for F in $BINPACKS; do
  if [ ! -f "$W/data/${F%.zst}.done" ]; then
    (curl -sSL "$HF/$F" | zstd -d -q -o "$W/data/${F%.zst}" --force &&
     touch "$W/data/${F%.zst}.done" &&
     echo "$(date +%H:%M:%S) pret : ${F%.zst} ($(du -h "$W/data/${F%.zst}" | cut -f1))" >> ~/setup_gpu.status) &
  fi
done

log "Alcyon (branche eval-tiers) et nnue-pytorch"
if [ ! -d "$W/alcyon" ]; then
  git clone -q -b eval-tiers https://github.com/EmericBraud/Alcyon.git "$W/alcyon"
fi
cd "$W/alcyon"
git submodule update --init -q training/nnue-pytorch
# Version qui a produit v3/v5 (voir nnue_model.hpp) : les plus recentes changent
# les signatures d'architecture et ne relisent plus v5.nnue.
(cd training/nnue-pytorch && git fetch -q origin 4289208fe20cc6ec8753e5ee14c2f210de783ff0 && git checkout -q 4289208fe20cc6ec8753e5ee14c2f210de783ff0)
cd "$W"

log "environnement Python (PyTorch CUDA 12.8 pour Blackwell)"
if [ ! -d "$W/venv" ]; then python3 -m venv "$W/venv"; fi
. "$W/venv/bin/activate"
pip install -q --no-cache-dir --upgrade pip
pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
REQ="$W/alcyon/training/nnue-pytorch/requirements.txt"
if [ -f "$REQ" ]; then pip install -q --no-cache-dir -r "$REQ" || true; fi
pip install -q --no-cache-dir lightning tyro
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'sm', torch.cuda.get_device_capability(0))" | tee -a ~/setup_gpu.status

log "chargeur de donnees de nnue-pytorch"
cd "$W/alcyon/training/nnue-pytorch"
if [ -f CMakeLists.txt ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j"$(nproc)" >/dev/null && log "chargeur compile (cmake)"
elif [ -f compile_data_loader.bat ]; then
  sh compile_data_loader.bat >/dev/null && log "chargeur compile (script)"
fi

log "moteur Alcyon"
cd "$W/alcyon"
CXX=g++-14 CC=gcc-14 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF >/dev/null && cmake --build build -j"$(nproc)" >/dev/null
(cd build && ALCYON_DATA_DIR=$PWD/data sh -c 'printf "bench\nquit\n" | ./alcyon | tail -1') | tee -a ~/setup_gpu.status

log "installation terminee (telechargements eventuellement encore en cours)"
