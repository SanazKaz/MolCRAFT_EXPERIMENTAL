#!/usr/bin/env python3
"""
Prepare BRD4_BD1 and CA-II targets for water-mediated pharmacophore guidance.

This script:
  1. Downloads crystal structures from the RCSB PDB (or accepts local files).
  2. Extracts the binding pocket around the reference ligand.
  3. Identifies conserved water positions (BRD4) or the zinc ion (CA-II).
  4. Computes the pocket centroid used for MolJO coordinate centering.
  5. Saves all metadata to <output_dir>/targets/<target>/<target>_guidance.json
     for use by sample_water_guided.py.

Usage
-----
    # Download structures automatically:
    python scripts/prepare_targets.py --output_dir data/targets

    # Or point at existing PDB files:
    python scripts/prepare_targets.py \\
        --brd4_pdb  data/raw/2OSS.pdb \\
        --caii_pdb  data/raw/1CA2.pdb \\
        --output_dir data/targets

Requirements
------------
    pip install biopython requests

Default structures
------------------
  BRD4 BD1:  PDB 2OSS  (BRD4 bromodomain + JQ1, 1.5 Å resolution)
             Waters in 2OSS are well-resolved; key conserved water at
             Asn140 (W1) is present across multiple bromodomain structures.
  CA-II:     PDB 1CA2  (human carbonic anhydrase II + acetazolamide)
             Catalytic Zn2+ is in the active site at the expected position.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
import numpy as np

# Allow running from the MolJO root or the scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.guidance.utils import (
    parse_waters_from_pdb,
    parse_metal_from_pdb,
    parse_pocket_protein_atoms,
    compute_pocket_centroid,
    transform_to_model_space,
    filter_waters_near_pocket,
    bfactor_to_weight,
)


# ---------------------------------------------------------------------------
# PDB download helper
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, output_path: str, overwrite: bool = False) -> str:
    """Download a PDB file from RCSB. Returns path to the file."""
    if os.path.exists(output_path) and not overwrite:
        print(f"  [cache] {output_path} already exists, skipping download.")
        return output_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Downloading {pdb_id} from {url} ...")
    urllib.request.urlretrieve(url, output_path)
    print(f"  Saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Pocket extraction (using Bio.PDB if available, else simple distance cut)
# ---------------------------------------------------------------------------

def extract_pocket_simple(
    pdb_path: str,
    ligand_resname: str,
    cutoff: float = 10.0,
    output_path: str = None,
) -> str:
    """
    Write a pocket PDB file containing protein residues within `cutoff` Å
    of any atom of the specified ligand residue.

    Requires no external tools; falls back to a simple distance-based filter.

    Args:
        pdb_path:      Full PDB file.
        ligand_resname:Three-letter residue name of the ligand (e.g. 'JQ1', 'AZM').
        cutoff:        Distance cutoff in Å.
        output_path:   Where to write the pocket PDB (auto-generated if None).

    Returns:
        Path to the written pocket PDB file.
    """
    if output_path is None:
        output_path = pdb_path.replace(".pdb", f"_pocket_{ligand_resname}.pdb")

    # Parse ligand positions
    ligand_pos = []
    protein_lines = []

    with open(pdb_path, "r") as fh:
        for line in fh:
            record = line[:6].strip()
            res_name = line[17:20].strip() if len(line) > 20 else ""
            if record == "HETATM" and res_name == ligand_resname:
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ligand_pos.append([x, y, z])
                except ValueError:
                    pass
            elif record == "ATOM":
                protein_lines.append(line)

    if not ligand_pos:
        raise ValueError(
            f"Ligand residue {ligand_resname} not found in {pdb_path}. "
            "Check the residue name in the PDB file."
        )
    ligand_arr = np.array(ligand_pos, dtype=np.float32)

    # Keep protein atoms within cutoff of any ligand atom
    pocket_lines = []
    for line in protein_lines:
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            pocket_lines.append(line)  # keep header-type lines
            continue
        pos = np.array([x, y, z], dtype=np.float32)
        dists = np.linalg.norm(ligand_arr - pos, axis=1)
        if dists.min() <= cutoff:
            pocket_lines.append(line)

    if not pocket_lines:
        raise ValueError(f"No protein atoms within {cutoff} Å of ligand {ligand_resname}.")

    with open(output_path, "w") as fh:
        fh.writelines(pocket_lines)
        fh.write("END\n")

    print(f"  Pocket written to {output_path} ({len(pocket_lines)} ATOM lines)")
    return output_path


# ---------------------------------------------------------------------------
# Target preparation functions
# ---------------------------------------------------------------------------

def prepare_brd4(
    pdb_path: str,
    output_dir: str,
    ligand_resname: str = "JQ1",
    pocket_cutoff: float = 10.0,
    max_bfactor: float = 60.0,
    water_pocket_radius: float = 10.0,
    pos_normalizer: float = 1.0,
) -> dict:
    """
    Prepare BRD4_BD1 target.

    Returns a dict with paths and metadata saved to <output_dir>/brd4_guidance.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract pocket
    pocket_path = os.path.join(output_dir, "brd4_pocket.pdb")
    try:
        pocket_path = extract_pocket_simple(pdb_path, ligand_resname, pocket_cutoff, pocket_path)
    except ValueError as e:
        # Try alternative ligand names found in BRD4 structures
        for alt_name in ["JQ1", "BRM", "LIG", "INH", "BI2"]:
            if alt_name == ligand_resname:
                continue
            print(f"  Trying alternative ligand name: {alt_name}")
            try:
                pocket_path = extract_pocket_simple(pdb_path, alt_name, pocket_cutoff, pocket_path)
                ligand_resname = alt_name
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Could not find ligand in {pdb_path}: {e}")

    # Compute pocket centroid from pocket protein atoms
    pocket_atoms = parse_pocket_protein_atoms(pocket_path)
    centroid = compute_pocket_centroid(pocket_atoms)
    print(f"  BRD4 pocket centroid: {centroid.tolist()}")

    # Extract water positions and B-factors from the FULL structure
    try:
        water_pos, bfactors = parse_waters_from_pdb(pdb_path)
    except ValueError:
        print("  WARNING: No waters found in full PDB; trying pocket file...")
        water_pos, bfactors = parse_waters_from_pdb(pocket_path)

    # Filter to waters near pocket
    water_pos_near, bfactors_near = filter_waters_near_pocket(
        water_pos, bfactors, centroid, radius=water_pocket_radius
    )
    weights = bfactor_to_weight(bfactors_near)
    water_model = transform_to_model_space(water_pos_near, centroid, pos_normalizer)

    print(
        f"  BRD4: {len(water_pos_near)} conserved waters "
        f"(B-factor ≤ {max_bfactor}, within {water_pocket_radius} Å)"
    )
    print(f"  Top-5 B-factors: {bfactors_near[:5].tolist()}")
    print(f"  Top-5 weights:   {weights[:5].tolist()}")

    guidance_data = {
        "target": "brd4_bd1",
        "pdb_id": os.path.basename(pdb_path).replace(".pdb", ""),
        "pocket_pdb": pocket_path,
        "full_pdb": pdb_path,
        "pocket_centroid": centroid.tolist(),
        "pos_normalizer": pos_normalizer,
        "guidance_type": "water_pharmacophore",
        "water_positions_crystal": water_pos_near.tolist(),
        "water_positions_model": water_model.tolist(),
        "bfactors": bfactors_near.tolist(),
        "weights": weights.tolist(),
        "n_pharmacophore_points": int(len(water_pos_near)),
        "params": {
            "sigma": 1.5,
            "offset": 0.5,
        },
    }

    out_json = os.path.join(output_dir, "brd4_guidance.json")
    with open(out_json, "w") as fh:
        json.dump(guidance_data, fh, indent=2)
    print(f"  BRD4 guidance data saved to {out_json}")

    return guidance_data


def prepare_caii(
    pdb_path: str,
    output_dir: str,
    ligand_resname: str = "AZM",
    pocket_cutoff: float = 10.0,
    pos_normalizer: float = 1.0,
) -> dict:
    """
    Prepare CA-II target.

    Returns a dict with paths and metadata saved to <output_dir>/caii_guidance.json.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract pocket
    pocket_path = os.path.join(output_dir, "caii_pocket.pdb")
    try:
        pocket_path = extract_pocket_simple(pdb_path, ligand_resname, pocket_cutoff, pocket_path)
    except ValueError as e:
        for alt_name in ["AZM", "ACE", "SO4", "ZNC", "LIG"]:
            if alt_name == ligand_resname:
                continue
            try:
                pocket_path = extract_pocket_simple(pdb_path, alt_name, pocket_cutoff, pocket_path)
                ligand_resname = alt_name
                break
            except ValueError:
                continue
        else:
            # Fall back: use zinc position as the pocket center
            print(
                "  WARNING: Could not find ligand; extracting pocket around zinc."
            )
            pocket_path = _pocket_around_metal(pdb_path, "ZN", pocket_cutoff, pocket_path)

    # Pocket centroid
    pocket_atoms = parse_pocket_protein_atoms(pocket_path)
    centroid = compute_pocket_centroid(pocket_atoms)
    print(f"  CA-II pocket centroid: {centroid.tolist()}")

    # Zinc position (from the full PDB)
    zinc_positions = parse_metal_from_pdb(pdb_path, element="ZN")
    zinc_crystal = zinc_positions[0]   # take first Zn
    zinc_model = transform_to_model_space(
        zinc_crystal.reshape(1, 3), centroid, pos_normalizer
    )[0]
    print(f"  Zinc crystal: {zinc_crystal.tolist()}")
    print(f"  Zinc model space: {zinc_model.tolist()}")

    guidance_data = {
        "target": "ca_ii",
        "pdb_id": os.path.basename(pdb_path).replace(".pdb", ""),
        "pocket_pdb": pocket_path,
        "full_pdb": pdb_path,
        "pocket_centroid": centroid.tolist(),
        "pos_normalizer": pos_normalizer,
        "guidance_type": "zinc_coordination",
        "zinc_position_crystal": zinc_crystal.tolist(),
        "zinc_position_model": zinc_model.tolist(),
        "params": {
            "d_target": 2.15 / pos_normalizer,
            "r_sigma": 0.3 / pos_normalizer,
            "n_target": 1.5,
            "n_sigma": 0.8,
            "w_coord": 1.0,
            "w_angle": 0.5,
            "ang_sigma": 0.3,
            "offset": 1.0,
        },
    }

    out_json = os.path.join(output_dir, "caii_guidance.json")
    with open(out_json, "w") as fh:
        json.dump(guidance_data, fh, indent=2)
    print(f"  CA-II guidance data saved to {out_json}")

    return guidance_data


def _pocket_around_metal(
    pdb_path: str, element: str, cutoff: float, output_path: str
) -> str:
    """Extract protein residues within cutoff Å of any metal atom."""
    metal_positions = parse_metal_from_pdb(pdb_path, element=element)
    if len(metal_positions) == 0:
        raise ValueError(f"No {element} found in {pdb_path}")

    protein_lines = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith("ATOM"):
                protein_lines.append(line)

    pocket_lines = []
    for line in protein_lines:
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        pos = np.array([x, y, z], dtype=np.float32)
        dists = np.linalg.norm(metal_positions - pos, axis=1)
        if dists.min() <= cutoff:
            pocket_lines.append(line)

    with open(output_path, "w") as fh:
        fh.writelines(pocket_lines)
        fh.write("END\n")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default="data/targets",
                        help="Root directory for target data")
    parser.add_argument("--brd4_pdb", default=None,
                        help="Path to BRD4 PDB file (downloads 2OSS if not given)")
    parser.add_argument("--caii_pdb", default=None,
                        help="Path to CA-II PDB file (downloads 1CA2 if not given)")
    parser.add_argument("--brd4_ligand", default="JQ1",
                        help="Residue name of the BRD4 ligand in the PDB (default: JQ1)")
    parser.add_argument("--caii_ligand", default="AZM",
                        help="Residue name of the CA-II ligand in the PDB (default: AZM)")
    parser.add_argument("--pocket_cutoff", type=float, default=10.0,
                        help="Å cutoff for pocket extraction around ligand")
    parser.add_argument("--max_bfactor", type=float, default=60.0,
                        help="BRD4: ignore waters with B-factor above this")
    parser.add_argument("--water_radius", type=float, default=10.0,
                        help="BRD4: water search radius around pocket centroid (Å)")
    parser.add_argument("--pos_normalizer", type=float, default=1.0,
                        help="MolJO position normalizer (cfg.data.normalizer_dict.pos)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download PDB files even if they exist")
    args = parser.parse_args()

    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # ---- BRD4_BD1 --------------------------------------------------------
    print("\n=== Preparing BRD4_BD1 (2OSS) ===")
    if args.brd4_pdb is None:
        brd4_pdb = download_pdb("2OSS", os.path.join(raw_dir, "2OSS.pdb"), args.overwrite)
    else:
        brd4_pdb = args.brd4_pdb

    brd4_out = os.path.join(args.output_dir, "brd4")
    brd4_data = prepare_brd4(
        pdb_path=brd4_pdb,
        output_dir=brd4_out,
        ligand_resname=args.brd4_ligand,
        pocket_cutoff=args.pocket_cutoff,
        max_bfactor=args.max_bfactor,
        water_pocket_radius=args.water_radius,
        pos_normalizer=args.pos_normalizer,
    )

    # ---- CA-II -----------------------------------------------------------
    print("\n=== Preparing CA-II (1CA2) ===")
    if args.caii_pdb is None:
        caii_pdb = download_pdb("1CA2", os.path.join(raw_dir, "1CA2.pdb"), args.overwrite)
    else:
        caii_pdb = args.caii_pdb

    caii_out = os.path.join(args.output_dir, "caii")
    caii_data = prepare_caii(
        pdb_path=caii_pdb,
        output_dir=caii_out,
        ligand_resname=args.caii_ligand,
        pocket_cutoff=args.pocket_cutoff,
        pos_normalizer=args.pos_normalizer,
    )

    print("\n=== Summary ===")
    print(f"BRD4 guidance:  {brd4_out}/brd4_guidance.json")
    print(f"  Pharmacophore points: {brd4_data['n_pharmacophore_points']}")
    print(f"CA-II guidance: {caii_out}/caii_guidance.json")
    print(f"  Zinc position (model): {caii_data['zinc_position_model']}")
    print("\nNext step: run scripts/sample_water_guided.py")


if __name__ == "__main__":
    main()
