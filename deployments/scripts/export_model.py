"""
export_model.py
───────────────
Chạy TRƯỚC khi `bentoml build` / `docker build`.
Mục đích:
  1. Kết nối MLflow Tracking Server
  2. Tải model @champion (version 1.0.0) về thư mục model_artifacts/
  3. Lưu metadata (run_id, version, artifact_uri) vào model_artifacts/meta.json
  4. Copy source code của model (nếu được log) vào model_artifacts/code/

Usage:
  python scripts/export_model.py \
      --tracking-uri http://localhost:5100 \
      --model-name   Churn_Predict \
      --alias        champion \
      --version      1.0.0 \
      --out-dir      ./model_artifacts
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# ─── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Export MLflow champion model for BentoML build")
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5100"))
    p.add_argument("--model-name",   default=os.getenv("MLFLOW_MODEL_NAME",   "Churn_Predict"))
    p.add_argument("--alias",        default="champion")
    p.add_argument("--version",      default="1.0.0",
                   help="Expected model version to validate against (semver string stored as tag)")
    p.add_argument("--out-dir",      default="./model_artifacts")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_version_by_alias(client: MlflowClient, model_name: str, alias: str):
    """Return the ModelVersion object pointed to by an alias."""
    mv = client.get_model_version_by_alias(model_name, alias)
    return mv


def validate_semver_tag(mv, expected_version: str):
    """
    We store the semver string as a tag 'semver' on the model version.
    e.g.  mlflow models tag  --model-name Churn_Predict --version 3 \
              --key semver --value 1.0.0
    """
    semver_tag = mv.tags.get("semver", "")
    if semver_tag != expected_version:
        print(
            f"[WARN] Model version tag 'semver'='{semver_tag}' "
            f"!= expected '{expected_version}'. Proceeding anyway."
        )
    else:
        print(f"[OK]   semver tag matches: {semver_tag}")
    return semver_tag or mv.version


def download_artifacts(client: MlflowClient, mv, out_dir: Path):
    """Download full artifact directory for this model version's run."""
    run_id = mv.run_id
    print(f"[INFO] Downloading artifacts for run_id={run_id} ...")
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        dst_path=str(out_dir / "mlflow_artifacts"),
    )
    print(f"[INFO] Artifacts saved to: {local_path}")
    return local_path


def download_model(model_uri: str, out_dir: Path):
    """Download the pyfunc model itself to a local directory."""
    dest = str(out_dir / "model")
    if Path(dest).exists():
        shutil.rmtree(dest)
    print(f"[INFO] Downloading model from {model_uri} → {dest}")
    mlflow.pyfunc.load_model(model_uri)          # warms up cache
    # Use mlflow.artifacts for an offline copy
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(out_dir),
    )
    print(f"[INFO] Model saved to: {dest}")


def copy_model_code(client: MlflowClient, mv, out_dir: Path):
    """
    If the training run logged source code under artifacts/code/,
    copy it so the Docker image can include it (for audit / retraining).
    """
    run = client.get_run(mv.run_id)
    artifacts = [a.path for a in client.list_artifacts(mv.run_id)]
    code_paths = [a for a in artifacts if a.startswith("code/")]
    if not code_paths:
        print("[INFO] No 'code/' artifacts found in run – skipping code copy.")
        return
    code_dir = out_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for cp in code_paths:
        local = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id,
            artifact_path=cp,
            dst_path=str(code_dir),
        )
        print(f"[INFO] Code artifact: {cp} → {local}")


def write_meta(mv, semver: str, out_dir: Path):
    meta = {
        "model_name":    mv.name,
        "mlflow_version": mv.version,
        "semver":         semver,
        "alias":          "champion",
        "run_id":         mv.run_id,
        "artifact_uri":   mv.source,
        "tags":           dict(mv.tags),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Metadata written to {meta_path}")
    return meta


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    print(f"[INFO] Tracking URI : {args.tracking_uri}")
    print(f"[INFO] Model        : {args.model_name}  alias=@{args.alias}")

    # 1. Resolve alias → ModelVersion
    mv = get_version_by_alias(client, args.model_name, args.alias)
    print(f"[INFO] Resolved alias @{args.alias} → version {mv.version}  run_id={mv.run_id}")

    # 2. Validate semver tag
    semver = validate_semver_tag(mv, args.version)

    # 3. Download model artifacts (offline copy)
    model_uri = f"models:/{args.model_name}@{args.alias}"
    download_model(model_uri, out_dir)

    # 4. Copy training source code if available
    copy_model_code(client, mv, out_dir)

    # 5. Write meta.json  (read by service.py at startup)
    meta = write_meta(mv, semver, out_dir)

    print("\n✅  Export complete:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
