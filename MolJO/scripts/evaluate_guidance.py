#!/usr/bin/env python3
"""
Evaluate guided vs. unguided MolJO outputs for BRD4_BD1 and CA-II.

Two evaluation metrics:
  1. **Vina Score** (vina_score mode) – fast scoring without re-docking.
     Requires AutoDock Vina installed and in PATH, or the MolJO vina wrapper.
  2. **ProLIF Interaction Fingerprints** – characterises:
       - Water-mediated hydrogen bonds (BRD4)
       - Zinc coordination contacts (CA-II)
       - General protein-ligand interactions

Usage
-----
    python scripts/evaluate_guidance.py \\
        --target brd4 \\
        --baseline_sdf  results/brd4_baseline/brd4_baseline_all.sdf \\
        --guided_sdf    results/brd4_guided/brd4_guided_all.sdf \\
        --protein_pdb   data/targets/raw/2OSS.pdb \\
        --pocket_pdb    data/targets/brd4/brd4_pocket.pdb \\
        --guidance_json data/targets/brd4/brd4_guidance.json \\
        --output_dir    results/eval_brd4

    python scripts/evaluate_guidance.py \\
        --target caii \\
        --baseline_sdf  results/caii_baseline/caii_baseline_all.sdf \\
        --guided_sdf    results/caii_guided/caii_guided_all.sdf \\
        --protein_pdb   data/targets/raw/1CA2.pdb \\
        --pocket_pdb    data/targets/caii/caii_pocket.pdb \\
        --guidance_json data/targets/caii/caii_guidance.json \\
        --output_dir    results/eval_caii

Requirements
------------
    pip install vina prolif MDAnalysis

Optional (for AutoDock Vina CLI fallback):
    conda install -c conda-forge autodock-vina
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Molecule loading helpers
# ---------------------------------------------------------------------------

def load_mols_from_sdf(sdf_path: str) -> List[Optional[Chem.Mol]]:
    """Load all molecules from an SDF file, returning None for failed entries."""
    suppl = SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    mols = []
    for mol in suppl:
        mols.append(mol)  # SDMolSupplier returns None on parse failure
    valid = sum(m is not None for m in mols)
    print(f"  Loaded {valid}/{len(mols)} valid molecules from {sdf_path}")
    return mols


def mol_to_sdf_string(mol: Chem.Mol) -> str:
    """Serialize a single molecule to SDF string."""
    from io import StringIO
    from rdkit.Chem import SDWriter
    buf = StringIO()
    w = SDWriter(buf)
    w.write(mol)
    w.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Vina scoring
# ---------------------------------------------------------------------------

def score_vina_python(
    mols: List[Optional[Chem.Mol]],
    protein_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
) -> List[Optional[float]]:
    """
    Score molecules using the Python Vina binding (vina package).

    Args:
        mols:          List of RDKit molecules.
        protein_pdb:   Path to the protein PDB (no waters/ligand).
        center:        Box center (x, y, z) in Angstrom (crystal coords).
        box_size:      Scoring box dimensions in Angstrom.
        exhaustiveness: Vina exhaustiveness parameter.

    Returns:
        List of Vina scores (kcal/mol) or None for failures.
    """
    try:
        from vina import Vina
    except ImportError:
        warnings.warn(
            "vina Python package not found. Install with: pip install vina\n"
            "Falling back to None scores."
        )
        return [None] * len(mols)

    try:
        from meeko import MoleculePreparation, PDBQTMolecule
    except ImportError:
        warnings.warn(
            "meeko not found (needed for Vina ligand preparation). "
            "Install with: pip install meeko"
        )
        return [None] * len(mols)

    # Prepare receptor PDBQT (done once)
    receptor_pdbqt = protein_pdb.replace(".pdb", "_receptor.pdbqt")
    if not os.path.exists(receptor_pdbqt):
        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(protein_pdb)
        v.write_pdbqt_file(receptor_pdbqt)

    scores = []
    for i, mol in enumerate(mols):
        if mol is None:
            scores.append(None)
            continue
        try:
            # Prepare ligand
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            lig_pdbqt = preparator.write_pdbqt_string()

            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(lig_pdbqt)
            v.compute_vina_maps(center=list(center), box_size=list(box_size))
            score = v.score()[0]
            scores.append(score)
        except Exception as e:
            scores.append(None)

    return scores


def score_vina_cli(
    mols: List[Optional[Chem.Mol]],
    protein_pdb: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    tmp_dir: str = "/tmp/eval_vina",
) -> List[Optional[float]]:
    """
    Score molecules using the AutoDock Vina CLI (fallback if Python API unavailable).
    """
    import subprocess
    import tempfile

    os.makedirs(tmp_dir, exist_ok=True)

    # Check Vina is available
    result = subprocess.run(["vina", "--version"], capture_output=True)
    if result.returncode != 0:
        warnings.warn("AutoDock Vina CLI not found in PATH.")
        return [None] * len(mols)

    scores = []
    for i, mol in enumerate(mols):
        if mol is None:
            scores.append(None)
            continue
        try:
            # Write ligand as SDF then convert to PDBQT using obabel
            lig_sdf = os.path.join(tmp_dir, f"lig_{i}.sdf")
            lig_pdbqt = os.path.join(tmp_dir, f"lig_{i}.pdbqt")
            out_pdbqt = os.path.join(tmp_dir, f"out_{i}.pdbqt")

            with Chem.SDWriter(lig_sdf) as w:
                w.write(mol)

            subprocess.run(
                ["obabel", lig_sdf, "-O", lig_pdbqt, "-h"],
                capture_output=True, check=True
            )

            cx, cy, cz = center
            sx, sy, sz = box_size
            result = subprocess.run(
                [
                    "vina",
                    "--receptor", protein_pdb.replace(".pdb", ".pdbqt"),
                    "--ligand", lig_pdbqt,
                    "--out", out_pdbqt,
                    "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                    "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                    "--scoring", "vina", "--score_only",
                ],
                capture_output=True, text=True
            )
            # Parse score from output
            for line in result.stdout.split("\n"):
                if "REMARK VINA RESULT" in line or "Affinity:" in line:
                    try:
                        score = float(line.split()[3])
                        scores.append(score)
                        break
                    except (IndexError, ValueError):
                        pass
            else:
                scores.append(None)
        except Exception as e:
            scores.append(None)

    return scores


def compute_vina_scores(
    mols: List[Optional[Chem.Mol]],
    protein_pdb: str,
    pocket_center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
) -> List[Optional[float]]:
    """Try Python Vina first, then fall back to CLI."""
    print("  Computing Vina scores...")
    scores = score_vina_python(mols, protein_pdb, pocket_center, box_size)
    n_none = sum(s is None for s in scores)
    if n_none > len(mols) * 0.5:
        print("  Python Vina largely failed; trying CLI fallback...")
        scores = score_vina_cli(mols, protein_pdb, pocket_center, box_size)
    return scores


# ---------------------------------------------------------------------------
# ProLIF interaction fingerprints
# ---------------------------------------------------------------------------

def compute_prolif_fingerprints(
    mols: List[Optional[Chem.Mol]],
    protein_pdb: str,
    water_positions: Optional[np.ndarray] = None,
    zinc_position: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute ProLIF interaction fingerprints for each molecule.

    Returns a DataFrame where rows are molecules and columns are interaction
    types (e.g. HBDonor, HBAcceptor, Hydrophobic, etc.).

    Also flags:
      - water_mediated_hbond: atom within 3.5 Å of a pharmacophore water
      - zinc_contact: coordinating atom within 2.5 Å of Zn
    """
    try:
        import prolif
        import MDAnalysis as mda
    except ImportError:
        warnings.warn(
            "prolif or MDAnalysis not installed. "
            "Install with: pip install prolif MDAnalysis\n"
            "Returning empty fingerprint DataFrame."
        )
        return pd.DataFrame({"error": ["prolif not installed"] * len(mols)})

    results = []

    try:
        u = mda.Universe(protein_pdb)
        protein_mol = prolif.Molecule.from_mda(u)
    except Exception as e:
        warnings.warn(f"Failed to load protein for ProLIF: {e}")
        return pd.DataFrame()

    for i, mol in enumerate(mols):
        row = {"mol_idx": i, "valid": mol is not None}

        if mol is None:
            results.append(row)
            continue

        try:
            lig = prolif.Molecule.from_rdkit(mol)
            fp = prolif.Fingerprint()
            fp.run_from_iterable([lig], protein_mol)
            df = fp.to_dataframe()
            if not df.empty:
                # Flatten multi-index columns
                interactions = {}
                for col in df.columns:
                    if isinstance(col, tuple):
                        key = f"{col[-1]}"
                    else:
                        key = str(col)
                    interactions[key] = bool(df[col].iloc[0])
                row.update(interactions)
        except Exception as e:
            row["prolif_error"] = str(e)

        # Custom contact checks
        if mol is not None:
            try:
                conf = mol.GetConformer()
                pos = conf.GetPositions()

                if water_positions is not None and len(water_positions) > 0:
                    min_water_dist = np.inf
                    for atom_pos in pos:
                        dists = np.linalg.norm(water_positions - atom_pos, axis=1)
                        min_water_dist = min(min_water_dist, dists.min())
                    row["min_water_dist_A"] = float(min_water_dist)
                    row["water_mediated_contact"] = bool(min_water_dist < 3.5)

                if zinc_position is not None:
                    zn = np.array(zinc_position)
                    # Check coordinating atom types (N, O, S)
                    coord_contacts = []
                    for atom in mol.GetAtoms():
                        sym = atom.GetSymbol()
                        if sym in ("N", "O", "S"):
                            atom_pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
                            d = np.linalg.norm(atom_pos - zn)
                            coord_contacts.append(d)
                    row["n_coord_atoms_in_shell"] = int(sum(d < 2.5 for d in coord_contacts))
                    row["min_coord_dist_A"] = float(min(coord_contacts)) if coord_contacts else np.nan
                    row["zinc_coordination_contact"] = bool(
                        any(d < 2.5 for d in coord_contacts)
                    )
            except Exception as e:
                row["contact_error"] = str(e)

        results.append(row)

    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# Basic molecular property calculations
# ---------------------------------------------------------------------------

def compute_mol_properties(mols: List[Optional[Chem.Mol]]) -> pd.DataFrame:
    """Compute basic molecular properties for validity assessment."""
    rows = []
    for i, mol in enumerate(mols):
        row = {"mol_idx": i, "valid": mol is not None}
        if mol is not None:
            try:
                row["mw"] = Descriptors.MolWt(mol)
                row["qed"] = Descriptors.qed(mol)
                row["n_hbd"] = rdMolDescriptors.CalcNumHBD(mol)
                row["n_hba"] = rdMolDescriptors.CalcNumHBA(mol)
                row["n_rotb"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                row["n_rings"] = rdMolDescriptors.CalcNumRings(mol)
                row["logp"] = Descriptors.MolLogP(mol)
                row["lipinski_ok"] = all([
                    row["mw"] <= 500,
                    row["logp"] <= 5,
                    row["n_hbd"] <= 5,
                    row["n_hba"] <= 10,
                ])
            except Exception as e:
                row["prop_error"] = str(e)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison reporting
# ---------------------------------------------------------------------------

def compare_and_report(
    target: str,
    baseline_results: dict,
    guided_results: dict,
    output_dir: str,
):
    """Generate comparison tables and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {target.upper()}")
    print("=" * 60)

    for label, results in [("Baseline", baseline_results), ("Guided", guided_results)]:
        print(f"\n--- {label} ---")
        mdf = results.get("mol_props")
        if mdf is not None and "valid" in mdf.columns:
            n_valid = mdf["valid"].sum()
            n_total = len(mdf)
            print(f"  Validity:     {n_valid}/{n_total} ({100*n_valid/max(n_total,1):.1f}%)")
            valid_mdf = mdf[mdf["valid"]]
            if not valid_mdf.empty:
                for col in ["mw", "qed", "logp"]:
                    if col in valid_mdf:
                        print(f"  {col:12s}: mean={valid_mdf[col].mean():.2f}, std={valid_mdf[col].std():.2f}")
                if "lipinski_ok" in valid_mdf:
                    print(f"  Lipinski ok:  {valid_mdf['lipinski_ok'].mean()*100:.1f}%")

        vina = results.get("vina_scores")
        if vina:
            valid_scores = [s for s in vina if s is not None]
            if valid_scores:
                print(f"  Vina score:   mean={np.mean(valid_scores):.2f} ± {np.std(valid_scores):.2f} kcal/mol")
                print(f"  Vina <-7:     {sum(s < -7 for s in valid_scores)/len(valid_scores)*100:.1f}%")

        pfdf = results.get("prolif_fp")
        if pfdf is not None and not pfdf.empty:
            if "water_mediated_contact" in pfdf.columns:
                rate = pfdf["water_mediated_contact"].mean()
                print(f"  Water contact rate: {rate*100:.1f}%")
            if "zinc_coordination_contact" in pfdf.columns:
                rate = pfdf["zinc_coordination_contact"].mean()
                print(f"  Zinc coord rate:    {rate*100:.1f}%")
            if "min_water_dist_A" in pfdf.columns:
                print(f"  Mean min water dist: {pfdf['min_water_dist_A'].mean():.2f} Å")
            if "min_coord_dist_A" in pfdf.columns:
                print(f"  Mean min Zn dist:   {pfdf['min_coord_dist_A'].mean():.2f} Å")
                print(f"  N coord atoms≤2.5Å: {pfdf['n_coord_atoms_in_shell'].mean():.2f}")

    # Save per-molecule tables
    for label, results in [("baseline", baseline_results), ("guided", guided_results)]:
        for key in ["mol_props", "prolif_fp"]:
            df = results.get(key)
            if df is not None and not df.empty:
                path = os.path.join(output_dir, f"{target}_{label}_{key}.csv")
                df.to_csv(path, index=False)
                print(f"\n  Saved {path}")

        vina = results.get("vina_scores")
        if vina:
            path = os.path.join(output_dir, f"{target}_{label}_vina.json")
            with open(path, "w") as fh:
                json.dump({"scores": vina}, fh)

    # Write combined summary JSON
    def safe_mean(lst):
        vals = [v for v in lst if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else None

    summary = {
        "target": target,
        "baseline": {
            "validity": float(baseline_results["mol_props"]["valid"].mean()) if baseline_results.get("mol_props") is not None else None,
            "vina_mean": safe_mean(baseline_results.get("vina_scores") or []),
            "water_contact_rate": float(baseline_results["prolif_fp"]["water_mediated_contact"].mean()) if baseline_results.get("prolif_fp") is not None and "water_mediated_contact" in baseline_results["prolif_fp"].columns else None,
            "zinc_contact_rate": float(baseline_results["prolif_fp"]["zinc_coordination_contact"].mean()) if baseline_results.get("prolif_fp") is not None and "zinc_coordination_contact" in baseline_results["prolif_fp"].columns else None,
        },
        "guided": {
            "validity": float(guided_results["mol_props"]["valid"].mean()) if guided_results.get("mol_props") is not None else None,
            "vina_mean": safe_mean(guided_results.get("vina_scores") or []),
            "water_contact_rate": float(guided_results["prolif_fp"]["water_mediated_contact"].mean()) if guided_results.get("prolif_fp") is not None and "water_mediated_contact" in guided_results["prolif_fp"].columns else None,
            "zinc_contact_rate": float(guided_results["prolif_fp"]["zinc_coordination_contact"].mean()) if guided_results.get("prolif_fp") is not None and "zinc_coordination_contact" in guided_results["prolif_fp"].columns else None,
        },
    }

    summary_path = os.path.join(output_dir, f"{target}_comparison.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nComparison summary: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, choices=["brd4", "caii"])
    parser.add_argument("--baseline_sdf", required=True,
                        help="SDF from baseline (no guidance) sampling")
    parser.add_argument("--guided_sdf", required=True,
                        help="SDF from guided sampling")
    parser.add_argument("--protein_pdb", required=True,
                        help="Full protein PDB (for Vina scoring)")
    parser.add_argument("--pocket_pdb", default=None,
                        help="Pocket PDB (optional, used for ProLIF)")
    parser.add_argument("--guidance_json", required=True,
                        help="Guidance metadata JSON from prepare_targets.py")
    parser.add_argument("--output_dir", default="results/eval",
                        help="Directory for evaluation outputs")
    parser.add_argument("--skip_vina", action="store_true",
                        help="Skip Vina scoring (e.g. if Vina not installed)")
    parser.add_argument("--skip_prolif", action="store_true",
                        help="Skip ProLIF fingerprinting")
    parser.add_argument("--vina_box_size", type=float, nargs=3, default=[20.0, 20.0, 20.0],
                        metavar=("X", "Y", "Z"),
                        help="Vina scoring box size in Angstrom")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load guidance metadata
    with open(args.guidance_json) as fh:
        guidance_data = json.load(fh)

    pocket_center_crystal = guidance_data["pocket_centroid"]
    water_positions = np.array(guidance_data.get("water_positions_crystal", []), dtype=np.float32) if "water_positions_crystal" in guidance_data else None
    zinc_position = np.array(guidance_data.get("zinc_position_crystal", []), dtype=np.float32) if "zinc_position_crystal" in guidance_data else None

    print(f"\n=== Loading molecules ===")
    baseline_mols = load_mols_from_sdf(args.baseline_sdf)
    guided_mols = load_mols_from_sdf(args.guided_sdf)

    print(f"\n=== Computing molecular properties ===")
    baseline_props = compute_mol_properties(baseline_mols)
    guided_props = compute_mol_properties(guided_mols)

    baseline_vina, guided_vina = [], []
    if not args.skip_vina:
        print(f"\n=== Computing Vina scores ===")
        print("  Baseline:")
        baseline_vina = compute_vina_scores(
            baseline_mols, args.protein_pdb,
            tuple(pocket_center_crystal), tuple(args.vina_box_size)
        )
        print("  Guided:")
        guided_vina = compute_vina_scores(
            guided_mols, args.protein_pdb,
            tuple(pocket_center_crystal), tuple(args.vina_box_size)
        )

    baseline_fp, guided_fp = pd.DataFrame(), pd.DataFrame()
    if not args.skip_prolif:
        print(f"\n=== Computing ProLIF fingerprints ===")
        pdb_for_prolif = args.pocket_pdb or args.protein_pdb
        print("  Baseline:")
        baseline_fp = compute_prolif_fingerprints(
            baseline_mols, pdb_for_prolif, water_positions, zinc_position
        )
        print("  Guided:")
        guided_fp = compute_prolif_fingerprints(
            guided_mols, pdb_for_prolif, water_positions, zinc_position
        )

    baseline_results = {
        "mol_props": baseline_props,
        "vina_scores": baseline_vina or None,
        "prolif_fp": baseline_fp if not baseline_fp.empty else None,
    }
    guided_results = {
        "mol_props": guided_props,
        "vina_scores": guided_vina or None,
        "prolif_fp": guided_fp if not guided_fp.empty else None,
    }

    summary = compare_and_report(args.target, baseline_results, guided_results, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
