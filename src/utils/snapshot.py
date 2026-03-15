"""Experiment snapshot utilities: create, list, and compare run snapshots."""

import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _git_hash():
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_tag(tag_name):
    """Create a git tag. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "tag", tag_name],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            logger.info(f"Created git tag: {tag_name}")
            return True
        else:
            logger.warning(f"Git tag failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.warning(f"Git tag failed: {e}")
        return False


def _zarr_fingerprint(zarr_path):
    """Hash zarr normalization stats as a quick identity fingerprint."""
    try:
        import numpy as np
        import zarr
        store = zarr.open(str(zarr_path), mode="r")
        parts = []
        for key in ["norm/target_mean", "norm/target_std", "norm/input_mean", "norm/input_std"]:
            if key in store:
                arr = np.array(store[key])
                parts.append(arr.tobytes())
        if parts:
            return hashlib.md5(b"".join(parts)).hexdigest()
    except Exception as e:
        logger.warning(f"Could not compute zarr fingerprint: {e}")
    return "unknown"


def create_snapshot(run_id, cfg, notes="", snapshot_base=None):
    """Create a snapshot of the current run state.

    Args:
        run_id: Identifier for this run (e.g. 'v1-baseline').
        cfg: Loaded ConfigNamespace with paths, training, etc.
        notes: Free-text description of the run.
        snapshot_base: Override snapshot directory (default: project_root/snapshots).
    """
    project_root = Path(cfg.paths.zarr_store).parent.parent
    snap_dir = Path(snapshot_base) if snapshot_base else project_root / "snapshots"
    run_dir = snap_dir / run_id

    if run_dir.exists():
        logger.error(f"Snapshot '{run_id}' already exists at {run_dir}. Aborting.")
        raise FileExistsError(f"Snapshot '{run_id}' already exists")

    run_dir.mkdir(parents=True)
    logger.info(f"Creating snapshot: {run_dir}")

    # 1. Copy config
    config_src = project_root / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, run_dir / "config.yaml")

    # 2. Copy best checkpoint
    ckpt_src = Path(cfg.paths.checkpoint_dir) / "best_model.pt"
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, run_dir / "best_model.pt")
        logger.info(f"  Copied checkpoint ({ckpt_src.stat().st_size / 1e6:.1f} MB)")
    else:
        logger.warning(f"  No checkpoint found at {ckpt_src}")

    # 3. Copy outputs (metrics, training history, ACF)
    output_dir = Path(cfg.paths.output_dir)
    for fname in ["metrics.json", "training_history.json", "acf_diagnostics.json"]:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, run_dir / fname)

    # 4. Copy spatial maps
    maps_src = output_dir / "spatial_maps"
    if maps_src.exists():
        maps_dst = run_dir / "spatial_maps"
        shutil.copytree(maps_src, maps_dst)
        n_files = len(list(maps_dst.iterdir()))
        logger.info(f"  Copied {n_files} spatial map files")

    # 5. Build manifest
    git_hash = _git_hash()
    zarr_md5 = _zarr_fingerprint(cfg.paths.zarr_store)

    # Load metrics summary if available
    metrics_summary = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_summary = json.load(f)

    # Load training history summary
    history_path = run_dir / "training_history.json"
    training_summary = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        if "val_loss" in history and history["val_loss"]:
            best_val = min(history["val_loss"])
            best_epoch = history["val_loss"].index(best_val) + 1
            training_summary = {
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "total_epochs": len(history["val_loss"]),
            }

    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_tag": run_id,
        "git_hash": git_hash,
        "notes": notes,
        "zarr_store": str(cfg.paths.zarr_store),
        "zarr_md5_meta": zarr_md5,
        "training_summary": training_summary,
        "metrics_summary": metrics_summary,
    }

    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Wrote manifest.json")

    # 6. Create git tag
    _git_tag(run_id)

    logger.info(f"Snapshot '{run_id}' complete at {run_dir}")
    return run_dir


def list_snapshots(snapshot_base=None, project_root=None):
    """List all snapshots with summary metrics.

    Args:
        snapshot_base: Path to snapshots directory.
        project_root: Project root (used to find default snapshot_base).

    Returns:
        List of manifest dicts.
    """
    if snapshot_base is None:
        if project_root is None:
            raise ValueError("Provide snapshot_base or project_root")
        snapshot_base = Path(project_root) / "snapshots"
    else:
        snapshot_base = Path(snapshot_base)

    if not snapshot_base.exists():
        print("No snapshots directory found.")
        return []

    snapshots = []
    for manifest_path in sorted(snapshot_base.glob("*/manifest.json")):
        with open(manifest_path) as f:
            manifest = json.load(f)
        snapshots.append(manifest)

    if not snapshots:
        print("No snapshots found.")
        return snapshots

    print(f"\n{'Run ID':<20} {'Date':<22} {'Git Hash':<10} {'Best Epoch':<12} {'Notes'}")
    print("-" * 90)
    for s in snapshots:
        epoch = s.get("training_summary", {}).get("best_epoch", "?")
        print(f"{s['run_id']:<20} {s['created']:<22} {s['git_hash']:<10} {str(epoch):<12} {s.get('notes', '')[:40]}")

    return snapshots


def compare_snapshots(id_a, id_b, snapshot_base=None, project_root=None):
    """Print side-by-side metrics comparison of two snapshots.

    Args:
        id_a: First run ID.
        id_b: Second run ID.
        snapshot_base: Path to snapshots directory.
        project_root: Project root (used to find default snapshot_base).
    """
    if snapshot_base is None:
        if project_root is None:
            raise ValueError("Provide snapshot_base or project_root")
        snapshot_base = Path(project_root) / "snapshots"
    else:
        snapshot_base = Path(snapshot_base)

    def _load(run_id):
        p = snapshot_base / run_id / "manifest.json"
        if not p.exists():
            raise FileNotFoundError(f"Snapshot '{run_id}' not found at {p}")
        with open(p) as f:
            return json.load(f)

    a = _load(id_a)
    b = _load(id_b)

    metrics_a = a.get("metrics_summary", {})
    metrics_b = b.get("metrics_summary", {})

    # Header
    print(f"\n{'':>20} {id_a:>16} {id_b:>16}   {'delta':>10}")
    print("=" * 70)

    # Training summary
    for key in ["best_val_loss", "best_epoch", "total_epochs"]:
        va = a.get("training_summary", {}).get(key, "?")
        vb = b.get("training_summary", {}).get(key, "?")
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            delta = f"{d:+.4f}" if isinstance(d, float) else f"{d:+d}"
        print(f"{key:>20} {str(va):>16} {str(vb):>16}   {delta:>10}")

    print("-" * 70)

    # Per-variable metrics
    variables = ["pet", "pck", "aet", "cwd"]
    for var in variables:
        ma = metrics_a.get(var, {})
        mb = metrics_b.get(var, {})
        all_keys = sorted(set(list(ma.keys()) + list(mb.keys())))
        for metric in all_keys:
            va = ma.get(metric)
            vb = mb.get(metric)
            label = f"{var.upper()} {metric}"
            sa = f"{va:.4f}" if isinstance(va, (int, float)) else "?"
            sb = f"{vb:.4f}" if isinstance(vb, (int, float)) else "?"
            delta = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = f"{vb - va:+.4f}"
            print(f"{label:>20} {sa:>16} {sb:>16}   {delta:>10}")

    # Zarr fingerprint comparison
    print("-" * 70)
    za = a.get("zarr_md5_meta", "?")
    zb = b.get("zarr_md5_meta", "?")
    same = "SAME" if za == zb else "DIFFERENT"
    print(f"{'zarr fingerprint':>20} {za[:16]:>16} {zb[:16]:>16}   {same:>10}")
    print()
