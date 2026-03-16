"""NSE distributions by L3 ecoregion — box/violin plots from snapshot spatial maps.

Usage:
    # Single snapshot
    python scripts/nse_by_ecoregion.py --snapshots v4-soil-props

    # Compare two snapshots side-by-side
    python scripts/nse_by_ecoregion.py --snapshots v3-vpd-awc v4-soil-props

    # Custom output path
    python scripts/nse_by_ecoregion.py --snapshots v4-soil-props -o outputs/eco_nse.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# EPA Level-3 ecoregion ID → short name
ECOREGION_NAMES = {
    1: "Coast Range",
    4: "Cascades",
    5: "Sierra Nevada",
    6: "C. Foothills",
    7: "Central Valley",
    8: "S. CA Mtns",
    9: "E. Cascades",
    13: "C. Basin & Range",
    14: "Mojave",
    78: "Klamath/N Coast",
    80: "N. Basin & Range",
    81: "Sonoran",
    85: "S. CA Coast",
}

VARIABLES = ["aet", "cwd", "pet", "pck"]


def load_nse_by_ecoregion(snap_dir, eco_raster, eco_nodata=-128):
    """Load NSE TIFs and group pixel values by ecoregion.

    Returns:
        dict[var][eco_id] → 1-D array of NSE values
    """
    snap_dir = Path(snap_dir)
    with rasterio.open(eco_raster) as src:
        eco = src.read(1)

    eco_ids = sorted(set(np.unique(eco)) - {eco_nodata})
    result = {}

    for var in VARIABLES:
        tif = snap_dir / "spatial_maps" / f"nse_{var}.tif"
        if not tif.exists():
            logger.warning(f"Missing {tif}")
            continue
        with rasterio.open(tif) as src:
            nse = src.read(1)
            nodata = src.nodata if src.nodata is not None else -9999.0

        valid = np.isfinite(nse) & (nse != nodata)
        result[var] = {}
        for eid in eco_ids:
            mask = valid & (eco == eid)
            if mask.sum() > 0:
                result[var][eid] = nse[mask]

    return result


def plot_nse_by_ecoregion(
    data_list,
    snap_names,
    eco_raster,
    output_path,
    eco_nodata=-128,
):
    """Generate box plots of NSE per ecoregion for 1 or 2 snapshots.

    Args:
        data_list: List of dicts from load_nse_by_ecoregion (1 or 2).
        snap_names: List of snapshot names matching data_list.
        eco_raster: Path to ecoregion raster (for ID list).
        output_path: Where to save the figure.
        eco_nodata: Nodata value for ecoregion raster.
    """
    n_snaps = len(data_list)
    colors = ["#4C72B0", "#DD8452"] if n_snaps == 2 else ["#4C72B0"]
    n_vars = len(VARIABLES)

    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars), constrained_layout=True)
    if n_vars == 1:
        axes = [axes]

    for ax, var in zip(axes, VARIABLES):
        # Collect all eco IDs present in any snapshot
        all_ids = set()
        for d in data_list:
            if var in d:
                all_ids.update(d[var].keys())
        if not all_ids:
            ax.set_title(f"{var.upper()} — no data")
            continue

        # Sort ecoregions by median NSE of first snapshot (descending)
        ref = data_list[0].get(var, {})
        eco_ids = sorted(all_ids, key=lambda eid: np.median(ref.get(eid, [0])), reverse=True)

        labels = [ECOREGION_NAMES.get(eid, str(eid)) for eid in eco_ids]
        positions = np.arange(len(eco_ids))

        for si, (d, name) in enumerate(zip(data_list, snap_names)):
            if var not in d:
                continue
            box_data = [d[var].get(eid, np.array([])) for eid in eco_ids]

            offset = -0.2 if n_snaps == 2 and si == 0 else (0.2 if n_snaps == 2 else 0.0)
            width = 0.35 if n_snaps == 2 else 0.6

            bp = ax.boxplot(
                box_data,
                positions=positions + offset,
                widths=width,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[si])
                patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("NSE")
        ax.set_title(f"{var.upper()} — NSE by Ecoregion")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
        ax.set_ylim(-1.5, 1.05)

        # Add pixel counts
        ref_data = data_list[0].get(var, {})
        for i, eid in enumerate(eco_ids):
            n = len(ref_data.get(eid, []))
            if n > 0:
                ax.text(i, -1.45, f"n={n}", ha="center", va="bottom", fontsize=7, color="gray")

    # Legend
    if n_snaps == 2:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], alpha=0.7, label=snap_names[0]),
            Patch(facecolor=colors[1], alpha=0.7, label=snap_names[1]),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ecoregion NSE plot: {output_path}")

    # Print summary table
    print(f"\n{'Ecoregion':<20}", end="")
    for name in snap_names:
        for var in VARIABLES:
            print(f" {name[:8]}_{var.upper():>4}", end="")
    print()
    print("-" * (20 + 14 * len(snap_names) * len(VARIABLES)))

    all_ids = set()
    for d in data_list:
        for var in VARIABLES:
            if var in d:
                all_ids.update(d[var].keys())
    eco_ids = sorted(all_ids)

    for eid in eco_ids:
        ename = ECOREGION_NAMES.get(eid, str(eid))
        print(f"{ename:<20}", end="")
        for d in data_list:
            for var in VARIABLES:
                vals = d.get(var, {}).get(eid, np.array([]))
                if len(vals) > 0:
                    print(f" {np.median(vals):>13.3f}", end="")
                else:
                    print(f" {'—':>13}", end="")
        print()


def generate_nse_by_ecoregion(
    snapshot_ids,
    project_root=".",
    eco_raster="/home/mmann1123/extra_space/Regions/ca_eco_l3.tif",
    output_path=None,
):
    """Public API: generate NSE-by-ecoregion plot from snapshot IDs.

    Args:
        snapshot_ids: List of 1 or 2 snapshot run IDs.
        project_root: Path to project root.
        eco_raster: Path to ecoregion raster.
        output_path: Output PNG path. Defaults to outputs/nse_by_ecoregion.png.

    Returns:
        Path to saved figure.
    """
    project_root = Path(project_root)
    snap_base = project_root / "snapshots"

    if output_path is None:
        output_path = project_root / "outputs" / "nse_by_ecoregion.png"

    data_list = []
    for sid in snapshot_ids:
        snap_dir = snap_base / sid
        if not snap_dir.exists():
            logger.warning(f"Snapshot '{sid}' not found at {snap_dir}, skipping")
            continue
        data_list.append(load_nse_by_ecoregion(snap_dir, eco_raster))

    valid_names = [sid for sid in snapshot_ids if (snap_base / sid).exists()]

    if not data_list:
        logger.error("No valid snapshots found")
        return None

    plot_nse_by_ecoregion(data_list, valid_names, eco_raster, output_path)
    return Path(output_path)


def main():
    parser = argparse.ArgumentParser(description="NSE by ecoregion box plots")
    parser.add_argument(
        "--snapshots", nargs="+", required=True,
        help="1 or 2 snapshot run IDs",
    )
    parser.add_argument(
        "--eco-raster",
        default="/home/mmann1123/extra_space/Regions/ca_eco_l3.tif",
        help="Path to L3 ecoregion raster",
    )
    parser.add_argument(
        "--project-root", default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: outputs/nse_by_ecoregion.png)",
    )
    args = parser.parse_args()

    generate_nse_by_ecoregion(
        snapshot_ids=args.snapshots,
        project_root=args.project_root,
        eco_raster=args.eco_raster,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
