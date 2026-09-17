#!/usr/bin/env bash
# arabic-suno-lora-finetuning — Colab bootstrap
#
# Run this BACKGROUNDED while you do `./code tunnel` auth in the foreground,
# so the auth wait and the install/download time overlap:
#
#   git clone https://github.com/akbargherbal/arabic-suno-lora-finetuning.git
#   cd arabic-suno-lora-finetuning
#   bash bootstrap/setup.sh > /content/logs/setup.log 2>&1 &
#   ../code tunnel
#
# Deliberately non-interactive — nothing here should prompt. HF_TOKEN is
# staged into a plain file by the notebook cell BEFORE this script runs, since
# `userdata` only exists inside the notebook kernel, not in a terminal shell.
# GCP auth itself happens separately via `gcloud auth login` in that same cell.
# GCP_DATASET_PATH falls back to this project's known prefix, so the dataset
# job also works if the notebook cell did not export it.
#
# Toolchain note: unlike yue2_lora_finetuning (a CLI trainer), this project
# trains through ComfyUI + filliptm/ComfyUI-FL-YuE2 graph nodes. FL-YuE2 can
# self-download its models at *queue* time via `download_missing`, but that
# costs VRAM-hours while you wait in the UI, so this script stages the same
# assets up front instead. Revisions below mirror FL-YuE2's own pins
# (yue2/downloads.py, yue2/training/downloads.py) so its in-node checksum
# verification passes on the first queue.

set -e
mkdir -p /content/logs

if [ -f /root/.secrets.env ]; then
  source /root/.secrets.env
else
  echo "[WARN] /root/.secrets.env not found — HF_TOKEN must already be in the environment"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[FAIL] HF_TOKEN is empty; stage /root/.secrets.env from the notebook cell first"
  exit 1
fi

# Anchor every ComfyUI path to the repo this script lives in, so the script
# behaves the same whether invoked from the repo root or elsewhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY="$REPO_ROOT/ComfyUI"
STAGING="/content/staging"

# --- Pinned revisions (match FL-YuE2's yue2/downloads.py + training/downloads.py) ---
YUE2_REV="1a96eca688d6ae5d7f0feb88573fec89920fcd19"       # m-a-p/YuE2-3B          (7.26 GB)
VAE_REV="95535e72a97bc0f09b8ada125d26b4009428c0e8"        # m-a-p/YuE2-Vae         (0.53 GB)
MERT_REV="d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"       # m-a-p/MERT-v2-FullSong (2.53 GB)
TOKENIZER_REV="f2278a2e005dc4ecc421c53a0929f62b3aeb2280"  # tokenizer head + NAR   (0.17 + 0.14 GB)
REGPACK_REV="5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35"    # minted regularizer     (0.10 GB)
# FL-YuE2 at the version this project's dataset_comfyui was verified against.
FLYUE2_REPO="https://github.com/filliptm/ComfyUI-FL-YuE2.git"
FLYUE2_COMMIT="80422124ef7a7b2132d4ae5da7cdfc7213a5c4f7"
# ComfyUI is intentionally NOT pinned: FL-YuE2 imports Comfy Kitchen's
# split-half RMS/RoPE op (comfy_kitchen.rms_rope_split_half) and needs a
# current ComfyUI. The resolved commit is printed by the verify step so the
# build this run used is still identifiable after the fact.
COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"

# `gsutil cp -r` nests the source folder under the destination, so this lands
# at /content/data/dataset/{dataset,dataset_comfyui}/.
DATASET_GCS="${GCP_DATASET_PATH:-gs://akbar-december-2024-backup/YuE2-3B_Arabic_Suno_Finetuning/dataset}"

echo "=== $(date) — arabic-suno bootstrap starting ==="

# --- Fast, synchronous: the `hf` CLI (>=0.34) and credentials ---
# Install rather than trust Colab's bundled huggingface_hub: `hf download`
# with --revision/--exclude needs a recent CLI, and it no longer prompts.
pip install -q --no-input "huggingface_hub>=0.36.0"
if ! command -v hf >/dev/null 2>&1; then
  echo "[FAIL] 'hf' CLI not found after installing huggingface_hub>=0.36 — check pip's bin dir is on PATH"
  exit 1
fi
hf auth login --token "$HF_TOKEN"

# --- Jobs (functions, so the background dispatch needs no quote gymnastics) ---

job_opencode() {
  curl -fsSL https://opencode.ai/install | bash
}

job_comfyui() {
  if [ ! -f "$COMFY/main.py" ]; then
    rm -rf "$COMFY" && git clone --quiet "$COMFY_REPO" "$COMFY"
  fi
  git -C "$COMFY" rev-parse --short HEAD
  pip install -q --no-input -r "$COMFY/requirements.txt"
}

job_dataset() {
  mkdir -p /content/data
  gsutil -m cp -r "$DATASET_GCS" /content/data
}

job_yue2() {
  hf download m-a-p/YuE2-3B --revision "$YUE2_REV" \
    --exclude "assets/*" --local-dir "$STAGING/models/yue2/YuE2-3B"
}

job_vae() {
  hf download m-a-p/YuE2-Vae --revision "$VAE_REV" \
    --exclude "assets/*" --local-dir "$STAGING/models/yue2/YuE2-Vae"
}

job_mert() {
  hf download m-a-p/MERT-v2-FullSong --revision "$MERT_REV" \
    --exclude "assets/*" --local-dir "$STAGING/models/yue2/MERT-v2-FullSong"
}

job_assets() {
  hf download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
    --revision "$TOKENIZER_REV" \
    tokenizer_head_joint_v4.pt nar_lora_joint_v4.pt \
    --local-dir "$STAGING/assets"
  hf download Mothersuperior/yue2-minted-corpus --repo-type dataset --revision "$REGPACK_REV" \
    regularizer/minted_regularizer_pack.pt \
    --local-dir "$STAGING/assets_regularizer"
}

# --- Launch every independent, slow task in parallel ---
pids=()
names=()

start_job() {
  local name="$1"; shift
  ("$@") > "/content/logs/${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

start_job opencode job_opencode
start_job comfyui job_comfyui
start_job dataset job_dataset
start_job yue2 job_yue2
start_job vae job_vae
start_job mert job_mert
start_job assets job_assets

echo "Launched in parallel: ${names[*]}"
echo "tail -f /content/logs/<name>.log to watch any one of these live."

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[ok]   ${names[$i]}"
  else
    echo "[FAIL] ${names[$i]} — see /content/logs/${names[$i]}.log"
    fail=1
  fi
done

# --- Merge the staged models into the now-cloned ComfyUI ---
# ComfyUI/the clone runs in parallel with the downloads; staging keeps the
# two from racing over $COMFY (and a re-clone cannot wipe 10 GB of weights).
merge_models() {
  local src_root="$STAGING/models/yue2"
  [ -d "$src_root" ] || { echo "[WARN] no staged models at $src_root"; return 0; }
  mkdir -p "$COMFY/models/yue2"
  local d name
  for d in "$src_root"/*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    rm -rf "$COMFY/models/yue2/$name"
    mv "$d" "$COMFY/models/yue2/$name"
  done
}

merge_assets() {
  local dst="$COMFY/models/yue2/training_assets"
  mkdir -p "$dst"
  local f
  for f in "$STAGING/assets/tokenizer_head_joint_v4.pt" \
           "$STAGING/assets/nar_lora_joint_v4.pt" \
           "$STAGING/assets_regularizer/regularizer/minted_regularizer_pack.pt"; do
    [ -f "$f" ] && mv -f "$f" "$dst/"
  done
}

echo "=== Merging staged downloads into ComfyUI ==="
merge_models
merge_assets

# --- FL-YuE2 custom node (depends on the ComfyUI clone above) ---
echo "Installing ComfyUI-FL-YuE2..."
mkdir -p "$COMFY/custom_nodes"
cd "$COMFY/custom_nodes"
if [ ! -d ComfyUI-FL-YuE2 ]; then
  git clone --quiet "$FLYUE2_REPO"
fi
git -C ComfyUI-FL-YuE2 fetch --quiet --all --tags
git -C ComfyUI-FL-YuE2 checkout --quiet "$FLYUE2_COMMIT"
# requirements-training.txt already `-r`s requirements.txt; installing it alone
# covers both, but name both so the intent survives a future edit of either file.
pip install -q --no-input \
  -r ComfyUI-FL-YuE2/requirements.txt \
  -r ComfyUI-FL-YuE2/requirements-training.txt

# --- Verify what actually landed, don't just trust exit codes ---
echo "=== Verifying downloads ==="

check_file() {  # path min_bytes label
  local path="$1" min="$2" label="$3"
  if [ ! -f "$path" ]; then
    echo "[FAIL] $label missing at $path"
    fail=1
    return 0
  fi
  local size
  size=$(stat -c%s "$path")
  if [ "$size" -lt "$min" ]; then
    echo "[FAIL] $label is only $size bytes — looks truncated, re-run the bootstrap"
    fail=1
  else
    echo "[ok]   $label $((size / 1024 / 1024)) MB"
  fi
}

check_file "$COMFY/models/yue2/YuE2-3B/model.safetensors" 7000000000 "YuE2-3B weights"
check_file "$COMFY/models/yue2/YuE2-Vae/model.safetensors" 500000000 "YuE2-Vae weights"
check_file "$COMFY/models/yue2/MERT-v2-FullSong/model.safetensors" 2400000000 "MERT-v2 weights"
check_file "$COMFY/models/yue2/training_assets/tokenizer_head_joint_v4.pt" 160000000 "tokenizer head"
check_file "$COMFY/models/yue2/training_assets/nar_lora_joint_v4.pt" 130000000 "NAR companion"
check_file "$COMFY/models/yue2/training_assets/minted_regularizer_pack.pt" 95000000 "minted regularizer"

# Dataset: gsutil nests, so FL-YuE2's flat training folder is
# /content/data/dataset/dataset_comfyui/ (267 tracks, 267 *.mp3).
track_count=$(find /content/data -path "*dataset_comfyui*" -name "*.mp3" 2>/dev/null | wc -l)
if [ "$track_count" -eq 0 ]; then
  echo "[FAIL] dataset looks empty — check /content/logs/dataset.log and the GCS path"
  fail=1
else
  echo "[ok]   dataset: $track_count tracks in /content/data/dataset/dataset_comfyui"
fi

if [ -f "$COMFY/custom_nodes/ComfyUI-FL-YuE2/__init__.py" ]; then
  fl_commit=$(git -C "$COMFY/custom_nodes/ComfyUI-FL-YuE2" rev-parse --short HEAD)
  echo "[ok]   ComfyUI-FL-YuE2 at $fl_commit"
else
  echo "[FAIL] ComfyUI-FL-YuE2 did not install"
  fail=1
fi
echo "[info] ComfyUI at $(git -C "$COMFY" rev-parse --short HEAD 2>/dev/null || echo '??') (tracking main, not pinned)"

# FL-YuE2 imports Comfy Kitchen's split-half RMS/RoPE op at model-load time.
# Warn here so a too-old ComfyUI surfaces during setup, not in the training run.
if (cd "$COMFY" && python -c "import comfy_kitchen" >/dev/null 2>&1); then
  echo "[ok]   comfy_kitchen importable"
else
  echo "[WARN] comfy_kitchen not importable — FL-YuE2 needs a current ComfyUI; update and re-run"
fi

if [ "$fail" -eq 1 ]; then
  echo "=== setup.sh finished WITH FAILURES — check the [FAIL] lines above ==="
  exit 1
fi

cat <<'EOF'
=== setup.sh done ===
Next (in a terminal):
  # 1. start ComfyUI (leave it running; no CLI training exists in FL-YuE2)
  setsid nohup python ComfyUI/main.py --listen 127.0.0.1 --port 8188 \
    < /dev/null > /content/logs/comfyui.log 2>&1 & disown
  #    wait until: curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8188/system_stats  ->  200
  # 2. open ComfyUI, load example_workflows/training_studio.json from the
  #    ComfyUI-FL-YuE2 custom node, point its Dataset Maker audio_directory at
  #    /content/data/dataset/dataset_comfyui, and queue.
EOF
