#!/usr/bin/env python3
"""
Sample molecules for BRD4_BD1 and CA-II with and without water-mediated
pharmacophore / zinc-coordination guidance.

This script runs MolJO sampling directly (bypassing the full PyTorch Lightning
trainer loop) for maximum control.  The MolJO model weights are kept frozen;
only the guidance signal at sampling time is modified.

Usage
-----
    # Prepare targets first:
    python scripts/prepare_targets.py --output_dir data/targets

    # Baseline (no guidance):
    python scripts/sample_water_guided.py \\
        --target brd4 \\
        --guidance_json data/targets/brd4/brd4_guidance.json \\
        --ckpt path/to/moljo.ckpt \\
        --config path/to/config.yaml \\
        --n_molecules 100 \\
        --no_guidance \\
        --output_dir results/brd4_baseline

    # With water pharmacophore guidance:
    python scripts/sample_water_guided.py \\
        --target brd4 \\
        --guidance_json data/targets/brd4/brd4_guidance.json \\
        --ckpt path/to/moljo.ckpt \\
        --config path/to/config.yaml \\
        --n_molecules 100 \\
        --pos_grad_weight 30 \\
        --type_grad_weight 0 \\
        --output_dir results/brd4_guided

    # CA-II with zinc coordination guidance:
    python scripts/sample_water_guided.py \\
        --target caii \\
        --guidance_json data/targets/caii/caii_guidance.json \\
        --ckpt path/to/moljo.ckpt \\
        --config path/to/config.yaml \\
        --n_molecules 100 \\
        --pos_grad_weight 20 \\
        --type_grad_weight 10 \\
        --output_dir results/caii_guided

Notes
-----
- protein_path / ligand_path in the guidance JSON are used to load the pocket
  and reference ligand for atom-count sampling.
- Output: <output_dir>/generated_<i>.sdf  and  <output_dir>/summary.json
"""

import argparse
import json
import os
import sys
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.transforms import Compose
from rdkit import Chem
from rdkit.Chem import SDWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.utils.transforms as trans
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
from core.models.train_loop import BFNTrainLoop
from core.config.config import Config
import core.utils.reconstruct as reconstruct
import core.utils.atom_num as atom_num

from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum


def load_model(ckpt_path: str, config_path: str, device: str) -> BFNTrainLoop:
    """
    Load MolJO model from checkpoint.

    Weights are NOT updated during guidance – this is sampling-time-only guidance.
    """
    cfg = Config(config_path)
    model = BFNTrainLoop.load_from_checkpoint(ckpt_path, config=cfg, map_location=device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"Loaded model from {ckpt_path}")
    return model, cfg


def build_dataloader(
    pocket_pdb: str,
    ligand_path: str,
    cfg: Config,
    n_molecules: int,
    batch_size: int = 4,
):
    """
    Build a dataloader from a pocket PDB and reference ligand SDF.

    The same pocket is replicated n_molecules times so the sampling loop
    runs independently for each molecule.
    """
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer])

    # Update feature dims in cfg so the model knows what to expect.
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim

    pocket = PDBProtein(pocket_pdb)
    pocket_dict = pocket.to_dict_atom()

    ligand_dict = parse_sdf_file(ligand_path)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = pocket_pdb
    data.ligand_filename = ligand_path
    data = transform(data)

    # Replicate: one copy per molecule to generate.
    dataset = [copy.deepcopy(data) for _ in range(n_molecules)]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    return loader


def build_guidance(guidance_data: dict, device: str, guidance_params_override: dict = None):
    """
    Instantiate the appropriate guidance object from the JSON metadata.

    Returns the guidance object or None (for baseline).
    """
    from core.guidance import WaterPharmacophoreGuidance, ZincCoordinationGuidance

    g_type = guidance_data["guidance_type"]
    params = dict(guidance_data.get("params", {}))
    if guidance_params_override:
        params.update(guidance_params_override)

    if g_type == "water_pharmacophore":
        guidance = WaterPharmacophoreGuidance(
            pharmacophore_positions=np.array(guidance_data["water_positions_model"], dtype=np.float32),
            weights=np.array(guidance_data["weights"], dtype=np.float32),
            sigma=params.get("sigma", 1.5),
            offset=params.get("offset", 0.5),
            device=device,
        )
        print(
            f"[Guidance] WaterPharmacophore: "
            f"{len(guidance_data['water_positions_model'])} points, "
            f"sigma={params.get('sigma', 1.5)}"
        )
    elif g_type == "zinc_coordination":
        guidance = ZincCoordinationGuidance(
            zinc_pos=np.array(guidance_data["zinc_position_model"], dtype=np.float32),
            d_target=params.get("d_target", 2.15),
            r_sigma=params.get("r_sigma", 0.3),
            n_target=params.get("n_target", 1.5),
            n_sigma=params.get("n_sigma", 0.8),
            w_coord=params.get("w_coord", 1.0),
            w_angle=params.get("w_angle", 0.5),
            ang_sigma=params.get("ang_sigma", 0.3),
            offset=params.get("offset", 1.0),
            device=device,
        )
        print(
            f"[Guidance] ZincCoordination: "
            f"Zn at {guidance_data['zinc_position_model']}, "
            f"d_target={params.get('d_target', 2.15):.3f}"
        )
    else:
        raise ValueError(f"Unknown guidance type: {g_type}")

    return guidance


def sample_molecules(
    model: BFNTrainLoop,
    cfg: Config,
    loader: DataLoader,
    device: str,
    classifiers=None,
    pos_grad_weight: float = 30.0,
    type_grad_weight: float = 10.0,
    sample_steps: int = 200,
    sample_num_atoms: str = "prior",
    guide_mode: str = "param_naive",
) -> list:
    """
    Run the MolJO sampling loop and return a list of RDKit molecules.

    Args:
        model:            Loaded BFNTrainLoop (weights frozen).
        cfg:              Config object.
        loader:           DataLoader over protein-ligand data.
        device:           Torch device.
        classifiers:      List of guidance objects or None for baseline.
        pos_grad_weight:  Gradient weight for coordinate guidance.
        type_grad_weight: Gradient weight for atom-type guidance.
        sample_steps:     Number of BFN denoising steps.
        sample_num_atoms: 'prior' (sample from prior) or 'ref' (use reference).
        guide_mode:       Guidance mode string.

    Returns:
        List of RDKit Mol objects (None entries for failed reconstructions).
    """
    from core.callbacks.validation_callback import center_pos

    dynamics = model.dynamics
    dynamics.eval()

    if classifiers is not None:
        model.configure_classifiers(
            classifiers=classifiers,
            objectives=[c.prop_name for c in classifiers],
            guide_mode=guide_mode,
            pos_grad_weight=pos_grad_weight,
            type_grad_weight=type_grad_weight,
        )
    else:
        model.classifiers = None
        model.guide_mode = None
        model.pos_grad_weight = pos_grad_weight
        model.type_grad_weight = type_grad_weight

    pos_normalizer = torch.tensor(
        cfg.data.normalizer_dict.pos, dtype=torch.float32, device=device
    )

    all_mols = []

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        print(f"  Batch {batch_idx+1}/{len(loader)}...")
        t0 = time.time()

        protein_pos = batch.protein_pos
        protein_v = batch.protein_atom_feature.float()
        batch_protein = batch.protein_element_batch

        # Apply normalizer
        protein_pos = protein_pos / pos_normalizer

        # Center protein to origin (same as MolJO training/validation)
        protein_pos, _, offset = center_pos(
            protein_pos, protein_pos, batch_protein, batch_protein,
            mode=cfg.dynamics.center_pos_mode
        )

        num_graphs = batch_protein.max().item() + 1

        # Determine number of ligand atoms to generate
        if sample_num_atoms == "prior":
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(
                    data.protein_pos.detach().cpu().numpy() * cfg.data.normalizer_dict.pos
                )
                ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size, cfg.data.path).astype(int))
            batch_ligand = torch.repeat_interleave(
                torch.arange(num_graphs, device=device),
                torch.tensor(ligand_num_atoms, device=device),
            )
            ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=device)
        elif sample_num_atoms == "ref":
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0)
        else:
            raise ValueError(f"Unsupported sample_num_atoms: {sample_num_atoms}")

        ligand_cum_atoms = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            ligand_num_atoms.cumsum(0),
        ])

        # ---- Core sampling call ----
        with torch.inference_mode(False):  # guidance requires grad
            theta_chain, sample_chain, y_chain = dynamics.sample(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                sample_steps=sample_steps,
                n_nodes=num_graphs,
                classifiers=model.classifiers,
                guide_mode=model.guide_mode,
                pos_grad_weight=model.pos_grad_weight,
                type_grad_weight=model.type_grad_weight,
                EPS=cfg.evaluation.get("interpolate_coef", 0.0),
                W_CFG=cfg.evaluation.get("cfg_coef", 0.0),
            )

        # ---- Reconstruct molecules ----
        final = sample_chain[-1]
        pred_pos = (final[0] + offset[batch_ligand]) * pos_normalizer
        one_hot = final[1]
        pred_v = one_hot.argmax(dim=-1)

        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=cfg.data.transform.ligand_atom_mode
        )
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=cfg.data.transform.ligand_atom_mode
        )
        pred_pos_np = pred_pos.detach().cpu().numpy().astype(np.float64)

        for i in range(num_graphs):
            start, end = int(ligand_cum_atoms[i]), int(ligand_cum_atoms[i + 1])
            try:
                mol = reconstruct.reconstruct_from_generated(
                    pred_pos_np[start:end],
                    pred_atom_type[start:end],
                    pred_aromatic[start:end],
                )
                all_mols.append(mol)
            except Exception as e:
                print(f"    Reconstruction failed for mol {i} in batch {batch_idx}: {e}")
                all_mols.append(None)

        print(f"  Batch {batch_idx+1} done in {time.time()-t0:.1f}s "
              f"({sum(m is not None for m in all_mols[-num_graphs:])} valid molecules)")

    return all_mols


def save_molecules(mols: list, output_dir: str, prefix: str = "generated") -> dict:
    """Save a list of RDKit molecules as individual SDF files and a combined SDF."""
    os.makedirs(output_dir, exist_ok=True)

    combined_path = os.path.join(output_dir, f"{prefix}_all.sdf")
    writer = SDWriter(combined_path)

    n_valid = 0
    saved_paths = []

    for i, mol in enumerate(mols):
        if mol is None:
            saved_paths.append(None)
            continue
        try:
            Chem.SanitizeMol(mol)
            mol_path = os.path.join(output_dir, f"{prefix}_{i:04d}.sdf")
            with Chem.SDWriter(mol_path) as w:
                w.write(mol)
            writer.write(mol)
            n_valid += 1
            saved_paths.append(mol_path)
        except Exception as e:
            print(f"  Failed to save mol {i}: {e}")
            saved_paths.append(None)

    writer.close()
    print(f"Saved {n_valid}/{len(mols)} valid molecules to {output_dir}/")
    print(f"Combined SDF: {combined_path}")

    return {
        "n_total": len(mols),
        "n_valid": n_valid,
        "combined_sdf": combined_path,
        "individual_sdfs": saved_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, choices=["brd4", "caii"],
                        help="Target identifier")
    parser.add_argument("--guidance_json", required=True,
                        help="Path to *_guidance.json from prepare_targets.py")
    parser.add_argument("--ckpt", required=True,
                        help="Path to MolJO checkpoint (.ckpt)")
    parser.add_argument("--config", required=True,
                        help="Path to the config YAML used during training")
    parser.add_argument("--n_molecules", type=int, default=100,
                        help="Number of molecules to generate")
    parser.add_argument("--no_guidance", action="store_true",
                        help="Run baseline sampling without any guidance")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save generated molecules")
    parser.add_argument("--pos_grad_weight", type=float, default=30.0,
                        help="Gradient weight for position guidance")
    parser.add_argument("--type_grad_weight", type=float, default=10.0,
                        help="Gradient weight for atom-type guidance")
    parser.add_argument("--sample_steps", type=int, default=200,
                        help="Number of BFN denoising steps")
    parser.add_argument("--sample_num_atoms", default="prior", choices=["prior", "ref"],
                        help="Atom count sampling strategy")
    parser.add_argument("--sampling_strategy", default="end_back_pmf",
                        help="BFN sampling strategy")
    parser.add_argument("--guide_mode", default="param_naive",
                        choices=["param_naive", "param_logit", "param_logit_2"],
                        help="Gradient guidance mode")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for sampling")
    parser.add_argument("--device", default=None,
                        help="Torch device (auto-detected if not specified)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Pharmacophore Gaussian sigma override (BRD4 only)")
    parser.add_argument("--offset", type=float, default=None,
                        help="Sigmoid offset override for guidance score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load guidance metadata
    with open(args.guidance_json) as fh:
        guidance_data = json.load(fh)

    pocket_pdb = guidance_data["pocket_pdb"]
    full_pdb = guidance_data["full_pdb"]
    pos_normalizer = guidance_data.get("pos_normalizer", 1.0)

    # A reference ligand SDF is needed for data loading.
    # Use the first SDF found near the pocket, or try to generate one from PDB HETATM.
    ligand_sdf = _find_or_create_ligand_sdf(full_pdb, pocket_pdb, args.target)

    # Load model
    model, cfg = load_model(args.ckpt, args.config, device)
    cfg.dynamics.sampling_strategy = args.sampling_strategy

    # Override normalizer from guidance metadata
    cfg.data.normalizer_dict.pos = pos_normalizer

    # Build dataloader
    loader = build_dataloader(
        pocket_pdb=pocket_pdb,
        ligand_path=ligand_sdf,
        cfg=cfg,
        n_molecules=args.n_molecules,
        batch_size=args.batch_size,
    )

    # Build guidance (or None for baseline)
    classifiers = None
    if not args.no_guidance:
        guidance_params_override = {}
        if args.sigma is not None:
            guidance_params_override["sigma"] = args.sigma
        if args.offset is not None:
            guidance_params_override["offset"] = args.offset

        guidance = build_guidance(guidance_data, device, guidance_params_override)
        guidance.prop_name = args.target  # label used in logging
        classifiers = [guidance]

    mode_label = "baseline" if args.no_guidance else "guided"
    print(f"\n=== Sampling {args.n_molecules} molecules ({mode_label}) for {args.target} ===")

    mols = sample_molecules(
        model=model,
        cfg=cfg,
        loader=loader,
        device=device,
        classifiers=classifiers,
        pos_grad_weight=args.pos_grad_weight,
        type_grad_weight=args.type_grad_weight,
        sample_steps=args.sample_steps,
        sample_num_atoms=args.sample_num_atoms,
        guide_mode=args.guide_mode,
    )

    # Save outputs
    out = save_molecules(mols, args.output_dir, prefix=f"{args.target}_{mode_label}")

    # Write run summary
    summary = {
        "target": args.target,
        "mode": mode_label,
        "guidance_json": args.guidance_json,
        "ckpt": args.ckpt,
        "n_molecules": args.n_molecules,
        "n_valid": out["n_valid"],
        "validity_rate": out["n_valid"] / max(args.n_molecules, 1),
        "pos_grad_weight": args.pos_grad_weight,
        "type_grad_weight": args.type_grad_weight,
        "sample_steps": args.sample_steps,
        "guide_mode": args.guide_mode,
        "combined_sdf": out["combined_sdf"],
    }
    summary_path = os.path.join(args.output_dir, f"{args.target}_{mode_label}_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary written to {summary_path}")
    print(f"Validity rate: {summary['validity_rate']:.1%}")


# ---------------------------------------------------------------------------
# Helper: find or create a reference ligand SDF for data loading
# ---------------------------------------------------------------------------

def _find_or_create_ligand_sdf(full_pdb: str, pocket_pdb: str, target: str) -> str:
    """
    Find an existing ligand SDF near the PDB file, or extract one from HETATM records.
    Returns path to an SDF file.
    """
    # Common ligand names for our two targets
    ligand_names = {
        "brd4": ["JQ1", "BRM", "BI2"],
        "caii": ["AZM", "ACE", "AMP"],
    }

    pdb_dir = os.path.dirname(full_pdb)
    pdb_stem = Path(full_pdb).stem

    # Check for existing SDF
    for pattern in [f"{pdb_stem}_lig*.sdf", "*.sdf", "ligand.sdf"]:
        import glob
        matches = glob.glob(os.path.join(pdb_dir, pattern))
        if matches:
            print(f"  Using existing ligand SDF: {matches[0]}")
            return matches[0]

    # Extract from PDB HETATM records
    print("  No ligand SDF found; extracting from PDB HETATM records...")
    candidates = ligand_names.get(target, ["LIG"])

    for resname in candidates:
        sdf_path = _extract_hetatm_to_sdf(full_pdb, resname, pdb_dir)
        if sdf_path is not None:
            print(f"  Extracted ligand {resname} to {sdf_path}")
            return sdf_path

    raise FileNotFoundError(
        f"Could not find or create a ligand SDF for {full_pdb}. "
        f"Please provide a ligand SDF file in the same directory as the PDB."
    )


def _extract_hetatm_to_sdf(pdb_path: str, resname: str, output_dir: str) -> str:
    """
    Use RDKit to extract a HETATM residue from a PDB file and save as SDF.
    Returns the SDF path or None if not found.
    """
    try:
        from rdkit.Chem import AllChem
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
        if mol is None:
            return None

        # Try to get the specific residue
        rwmol = Chem.RWMol(mol)
        atoms_to_keep = []
        for atom in rwmol.GetAtoms():
            mi = atom.GetMonomerInfo()
            if mi is not None and mi.GetResidueName().strip() == resname:
                atoms_to_keep.append(atom.GetIdx())

        if not atoms_to_keep:
            return None

        out_path = os.path.join(output_dir, f"{resname}_lig.sdf")
        em = Chem.RWMol()
        idx_map = {}
        for old_idx in atoms_to_keep:
            atom = rwmol.GetAtomWithIdx(old_idx)
            new_idx = em.AddAtom(atom)
            idx_map[old_idx] = new_idx

        conf = Chem.Conformer(len(atoms_to_keep))
        orig_conf = mol.GetConformer()
        for old_idx, new_idx in idx_map.items():
            pos = orig_conf.GetAtomPosition(old_idx)
            conf.SetAtomPosition(new_idx, pos)
        em.AddConformer(conf, assignId=True)

        try:
            Chem.SanitizeMol(em)
        except Exception:
            pass

        with Chem.SDWriter(out_path) as w:
            w.write(em)
        return out_path

    except Exception as e:
        print(f"  Failed to extract {resname} from PDB: {e}")
        return None


if __name__ == "__main__":
    main()
