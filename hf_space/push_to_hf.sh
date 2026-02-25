#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  push_to_hf.sh — Clone the HF Space repo, populate it with model artifacts,
#                  and push everything to Hugging Face.
#
#  Prerequisites:
#    - git + git-lfs installed  (brew install git-lfs)
#    - HF token with WRITE access set in HF_TOKEN env var, OR you'll be
#      prompted for credentials (use your HF token as the password).
#
#  Usage (from repo root):
#    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#    bash hf_space/push_to_hf.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_SPACE_DIR="$REPO_ROOT/hf_space"
CLONE_DIR="$REPO_ROOT/_hf_space_clone"
HF_SPACE_ID="PietroSaveri/Selene_ML_APIs"
HF_SPACE_URL="https://huggingface.co/spaces/$HF_SPACE_ID"

echo "▶ Repo root:   $REPO_ROOT"
echo "▶ HF Space:    $HF_SPACE_URL"
echo ""

# ── 1. Ensure git-lfs is available ───────────────────────────────────────────
if ! command -v git-lfs &>/dev/null; then
    echo "❌  git-lfs not found. Install it first:  brew install git-lfs"
    exit 1
fi
git lfs install --skip-smudge

# ── 2. Clone the HF Space repo ────────────────────────────────────────────────
if [ -d "$CLONE_DIR" ]; then
    echo "ℹ  Clone directory already exists — pulling latest …"
    cd "$CLONE_DIR" && git pull
else
    echo "▶ Cloning $HF_SPACE_URL …"
    if [ -n "${HF_TOKEN:-}" ]; then
        # Embed token in URL so it doesn't prompt
        AUTHED_URL="https://pietrosaveri:${HF_TOKEN}@huggingface.co/spaces/$HF_SPACE_ID"
        git clone "$AUTHED_URL" "$CLONE_DIR"
    else
        git clone "$HF_SPACE_URL" "$CLONE_DIR"
    fi
fi

cd "$CLONE_DIR"

# ── 3. Copy app files from hf_space/ ─────────────────────────────────────────
echo "▶ Copying app files …"
cp "$HF_SPACE_DIR/Dockerfile"       ./Dockerfile
cp "$HF_SPACE_DIR/serve.py"         ./serve.py
cp "$HF_SPACE_DIR/requirements.txt" ./requirements.txt
cp "$HF_SPACE_DIR/README.md"        ./README.md

# ── 4. Copy clustering artifacts ─────────────────────────────────────────────
echo "▶ Copying clustering artifacts …"
mkdir -p artifacts/clustering
cp "$REPO_ROOT/models/clustering/artifacts/gmm_model.pkl"       artifacts/clustering/
cp "$REPO_ROOT/models/clustering/artifacts/scaler.pkl"          artifacts/clustering/
cp "$REPO_ROOT/models/clustering/artifacts/imputer.pkl"         artifacts/clustering/
cp "$REPO_ROOT/models/clustering/artifacts/profile_rules.json"  artifacts/clustering/

# ── 5. Copy simulation artifacts ─────────────────────────────────────────────
echo "▶ Copying simulation artifacts …"
mkdir -p artifacts/simulation
cp "$REPO_ROOT/models/simulation/artifacts/model_symptoms.pkl"     artifacts/simulation/
cp "$REPO_ROOT/models/simulation/artifacts/model_satisfaction.pkl" artifacts/simulation/
cp "$REPO_ROOT/models/simulation/artifacts/feature_meta.json"      artifacts/simulation/

# ── 6. Copy pill reference DB ────────────────────────────────────────────────
echo "▶ Copying pill reference DB …"
mkdir -p drugs/output
cp "$REPO_ROOT/drugs/output/pill_reference_db.csv" drugs/output/

# ── 7. Track large binary files with LFS ─────────────────────────────────────
echo "▶ Setting up git-lfs tracking …"
git lfs track "*.pkl"
git add .gitattributes 2>/dev/null || true

# ── 8. Stage and commit ───────────────────────────────────────────────────────
echo "▶ Staging files …"
git add -A

if git diff --cached --quiet; then
    echo "ℹ  Nothing to commit — Space is already up to date."
else
    echo "▶ Committing …"
    git commit -m "Deploy combined cluster + simulator APIs (FastAPI / HF Docker Space)"

    echo "▶ Pushing to Hugging Face …"
    if [ -n "${HF_TOKEN:-}" ]; then
        # Re-set remote with token in URL for the push
        git remote set-url origin "https://pietrosaveri:${HF_TOKEN}@huggingface.co/spaces/$HF_SPACE_ID"
    fi
    git push origin main

    echo "✅  Deployed!  Space will rebuild at:"
    echo "    $HF_SPACE_URL"
fi
