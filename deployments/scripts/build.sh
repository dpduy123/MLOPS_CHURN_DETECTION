#!/usr/bin/env bash
# scripts/build.sh
# Usage: ./scripts/build.sh [--push]
# ─────────────────────────────────────────────────────────────────────────────
# 1. Export champion model từ MLflow (offline copy + meta.json)
# 2. bentoml build
# 3. bentoml containerize  → Docker image
# 4. (optional) docker push
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BENTO_DIR="$ROOT_DIR/bentoml"
ARTIFACTS_DIR="$ROOT_DIR/model_artifacts"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5100}"
MODEL_NAME="${MODEL_NAME:-Churn_Predict}"
MODEL_ALIAS="${MODEL_ALIAS:-champion}"
MODEL_VERSION="${MODEL_VERSION:-1.0.0}"
IMAGE_TAG="${IMAGE_TAG:-churn-prediction:latest}"

echo "════════════════════════════════════════════════════════"
echo "  MLOps Build Pipeline — Churn Prediction"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Export model ──────────────────────────────────────────────────────
echo ""
echo "▶ [1/4] Exporting model @${MODEL_ALIAS} (v${MODEL_VERSION}) from MLflow..."
python "$SCRIPT_DIR/export_model.py" \
  --tracking-uri "$MLFLOW_TRACKING_URI" \
  --model-name   "$MODEL_NAME" \
  --alias        "$MODEL_ALIAS" \
  --version      "$MODEL_VERSION" \
  --out-dir      "$ARTIFACTS_DIR"

echo "✓ Model exported to $ARTIFACTS_DIR"

# ── Step 2: Copy artifacts into bentoml/ so bentofile.yaml can include them ──
echo ""
echo "▶ [2/4] Copying artifacts into bentoml build context..."
cp -r "$ARTIFACTS_DIR" "$BENTO_DIR/model_artifacts"
echo "✓ Artifacts copied"

# ── Step 3: bentoml build ─────────────────────────────────────────────────────
echo ""
echo "▶ [3/4] Running bentoml build..."
cd "$BENTO_DIR"
bentoml build -f bentofile.yaml .
BENTO_TAG=$(bentoml list churn-prediction --output json | python3 -c \
  "import sys,json; items=json.load(sys.stdin); print(items[0]['tag'])")
echo "✓ Bento built: $BENTO_TAG"

# ── Step 4: containerize ─────────────────────────────────────────────────────
echo ""
echo "▶ [4/4] Containerizing → $IMAGE_TAG ..."
bentoml containerize "$BENTO_TAG" -t "$IMAGE_TAG"
echo "✓ Docker image: $IMAGE_TAG"

# ── Optional push ─────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--push" ]]; then
  echo ""
  echo "▶ Pushing $IMAGE_TAG ..."
  docker push "$IMAGE_TAG"
  echo "✓ Pushed"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Build complete ✅"
echo "  Image: $IMAGE_TAG"
echo "  Run:   docker-compose up -d"
echo "════════════════════════════════════════════════════════"
