#!/usr/bin/env python3
"""
Water-mediated pharmacophore and zinc coordination guidance for MolJO.

This script is a thin wrapper around MolJO's own sample_for_pocket_guided.py.
It replaces only the classifier injection — all sampling, normalisation,
centering, and reconstruction use MolJO's unmodified code.

Experimental conditions
-----------------------
  BASELINE (no guidance):
      Use MolCRAFT/sample_for_pocket.py directly — the pure backbone model.
      Do NOT use this script for the baseline.

  GUIDED (water pharmacophore / zinc coordination):
      python MolJO/scripts/sample_water_guided.py  [args below]

  AFFINITY GUIDED (MolJO's own classifiers):
      Use MolJO/sample_for_pocket_guided.py with --objective vina_sa

Usage
-----
    python MolJO/scripts/sample_water_guided.py \\
        --config_file  MolJO/configs/test_opt.yaml \\
        --protein_path data/targets/raw/4WHW.pdb \\
        --ligand_path  data/targets/raw/4WHW_lig.sdf \\
        --guidance_json data/targets/brd4/brd4_guidance.json \\
        --ckpt         /path/to/moljo.ckpt \\
        --num_samples  100 \\
        --pos_grad_weight 30 \\
        --type_grad_weight 0 \\
        --no_wandb

    # CA-II with zinc coordination:
    python MolJO/scripts/sample_water_guided.py \\
        --config_file  MolJO/configs/test_opt.yaml \\
        --protein_path data/targets/raw/6EX1.pdb \\
        --ligand_path  data/targets/raw/6EX1_lig.sdf \\
        --guidance_json data/targets/caii/caii_guidance.json \\
        --ckpt         /path/to/moljo.ckpt \\
        --num_samples  100 \\
        --pos_grad_weight 20 \\
        --type_grad_weight 10 \\
        --no_wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── MolJO infrastructure (unmodified) ─────────────────────────────────────────
from core.config.config import Config
from core.models.train_loop import BFNTrainLoop
from core.callbacks.basic import RecoverCallback, NormalizerCallback, EMACallback, GradientClip
from core.callbacks.validation_callback import (
    CondMolGenValidationCallback,
    MolVisualizationCallback,
    ReconValidationCallback,
    DockingTestCallback,
)
from core.datasets.utils import PDBProtein, parse_sdf_file
from core.datasets.pl_data import ProteinLigandData, torchify_dict, FOLLOW_BATCH
import core.utils.transforms as trans

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# ── Our guidance (injection only) ─────────────────────────────────────────────
from core.guidance import WaterPharmacophoreGuidance, ZincCoordinationGuidance


def get_dataloader_from_pdb(cfg):
    """
    Identical to MolJO's sample_for_pocket_guided.py::get_dataloader_from_pdb.
    Kept here so this script is self-contained.
    """
    protein_fn = cfg.evaluation.protein_path
    ligand_fn  = cfg.evaluation.ligand_path

    protein = PDBProtein(protein_fn)
    ligand_dict = parse_sdf_file(ligand_fn)
    pdb_block_pocket = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand_dict, cfg.dynamics.net_config.r_max)
    )
    pocket = PDBProtein(pdb_block_pocket)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket.to_dict_atom()),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename   = ligand_fn

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer  = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer])
    cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
    cfg.dynamics.ligand_atom_feature_dim  = ligand_featurizer.feature_dim

    collate_exclude_keys = ["ligand_nbh_list"]
    test_set = [transform(data)] * cfg.evaluation.num_samples
    cfg.evaluation.num_samples = 1
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )
    cfg.evaluation.docking_config.protein_root = os.path.dirname(
        os.path.abspath(protein_fn)
    )
    return test_loader


def build_guidance(guidance_json_path: str, device: str = "cpu"):
    """
    Instantiate the appropriate guidance object from the JSON metadata
    produced by prepare_targets.py.
    """
    with open(guidance_json_path) as fh:
        gdata = json.load(fh)

    g_type = gdata["guidance_type"]
    params = gdata.get("params", {})

    if g_type == "water_pharmacophore":
        guidance = WaterPharmacophoreGuidance(
            pharmacophore_positions=np.array(gdata["water_positions_model"], dtype=np.float32),
            weights=np.array(gdata["weights"], dtype=np.float32),
            sigma=params.get("sigma", 1.5),
            offset=params.get("offset", 0.5),
            device=device,
        )
        print(
            f"[Guidance] WaterPharmacophore — "
            f"{len(gdata['water_positions_model'])} points, "
            f"sigma={params.get('sigma', 1.5)}"
        )
    elif g_type == "zinc_coordination":
        guidance = ZincCoordinationGuidance(
            zinc_pos=np.array(gdata["zinc_position_model"], dtype=np.float32),
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
            f"[Guidance] ZincCoordination — "
            f"Zn at {gdata['zinc_position_model']}, "
            f"d_target={params.get('d_target', 2.15):.3f} Å"
        )
    else:
        raise ValueError(f"Unknown guidance type in JSON: {g_type}")

    return guidance, gdata["target"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Config / checkpoint — mirrors MolJO's own script
    parser.add_argument("--config_file", default="configs/test_opt.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to MolJO .ckpt file")
    parser.add_argument("--exp_name",  default="water_guided")
    parser.add_argument("--revision",  default="default")
    parser.add_argument("--no_wandb",  action="store_true")
    parser.add_argument("--seed",      type=int, default=1234)

    # Target inputs
    parser.add_argument("--protein_path", required=True)
    parser.add_argument("--ligand_path",  required=True,
                        help="Reference ligand SDF (used for pocket extraction and atom-count prior)")

    # Our guidance
    parser.add_argument("--guidance_json", required=True,
                        help="JSON from prepare_targets.py")

    # Sampling hyperparameters
    parser.add_argument("--num_samples",      type=int,   default=100)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--sample_steps",     type=int,   default=200)
    parser.add_argument("--sample_num_atoms", default="prior",
                        choices=["prior", "ref"])
    parser.add_argument("--sampling_strategy",default="end_back_pmf")
    parser.add_argument("--pos_normalizer",   type=float, default=1.0)

    # Guidance weights
    parser.add_argument("--guide_mode",        default="param_naive",
                        choices=["param_naive", "param_logit", "param_logit_2"])
    parser.add_argument("--pos_grad_weight",   type=float, default=30.0)
    parser.add_argument("--type_grad_weight",  type=float, default=10.0)

    # BFN params (keep defaults matching the checkpoint)
    parser.add_argument("--sigma1_coord",  type=float, default=0.03)
    parser.add_argument("--beta1",         type=float, default=1.5)
    parser.add_argument("--t_min",         type=float, default=0.0001)
    parser.add_argument("--docking_mode",  default="vina_score")

    args = parser.parse_args()
    seed_everything(args.seed)

    # ── Load MolJO config ──────────────────────────────────────────────────────
    cfg = Config(args.config_file,
                 exp_name=args.exp_name,
                 revision=args.revision,
                 no_wandb=args.no_wandb,
                 seed=args.seed,
                 test_only=True,
                 debug=False,
                 resume=False,
                 wandb_resume_id=None,
                 empty_folder=False,
                 logging_level="warning",
                 random_rot=False,
                 pos_noise_std=0,
                 pos_normalizer=args.pos_normalizer,
                 batch_size=args.batch_size,
                 epochs=1,
                 v_loss_weight=1,
                 lr=5e-4,
                 scheduler="plateau",
                 weight_decay=0,
                 max_grad_norm="Q",
                 sigma1_coord=args.sigma1_coord,
                 beta1=args.beta1,
                 t_min=args.t_min,
                 use_discrete_t=True,
                 discrete_steps=1000,
                 destination_prediction=True,
                 sampling_strategy=args.sampling_strategy,
                 time_emb_mode="simple",
                 time_emb_dim=0,
                 pos_init_mode="zero",
                 num_samples=args.num_samples,
                 sample_steps=args.sample_steps,
                 sample_num_atoms=args.sample_num_atoms,
                 visual_chain=False,
                 protein_path=args.protein_path,
                 ligand_path=args.ligand_path,
                 last_ckpt=False,
                 docking_mode=args.docking_mode,
                 save_traj=False,
                 objective=None,        # no MolJO affinity classifiers
                 guide_mode=args.guide_mode,
                 pos_grad_weight=args.pos_grad_weight,
                 type_grad_weight=args.type_grad_weight,
                 interpolate_coef=0.0,
                 )

    # Load the training config saved alongside the checkpoint
    ckpt_config = os.path.join(os.path.dirname(args.ckpt), "..", "config.yaml")
    if os.path.exists(ckpt_config):
        tr_cfg = Config(ckpt_config)
        tr_cfg.test_only        = cfg.test_only
        tr_cfg.evaluation       = cfg.evaluation
        tr_cfg.visual           = cfg.visual
        tr_cfg.accounting       = cfg.accounting
        tr_cfg.dynamics.beta1            = cfg.dynamics.beta1
        tr_cfg.dynamics.sigma1_coord     = cfg.dynamics.sigma1_coord
        tr_cfg.dynamics.sampling_strategy = cfg.dynamics.sampling_strategy
        tr_cfg.seed = cfg.seed
        if not hasattr(tr_cfg.train, "max_grad_norm"):
            tr_cfg.train.max_grad_norm = "Q"
        cfg = tr_cfg
        print(f"Loaded training config from {ckpt_config}")
    else:
        print(f"No saved training config found at {ckpt_config}; using CLI config.")

    # ── Data ───────────────────────────────────────────────────────────────────
    test_loader = get_dataloader_from_pdb(cfg)

    # ── Model (MolJO, weights frozen) ─────────────────────────────────────────
    model = BFNTrainLoop(config=cfg)

    # ── Inject our guidance classifiers ───────────────────────────────────────
    guidance, target_name = build_guidance(args.guidance_json)
    guidance.prop_name = target_name
    model.configure_classifiers(
        classifiers=[guidance],
        objectives=[target_name],
        guide_mode=args.guide_mode,
        pos_grad_weight=args.pos_grad_weight,
        type_grad_weight=args.type_grad_weight,
    )
    print(
        f"\n[Setup] Guidance: {target_name} | "
        f"pos_grad_weight={args.pos_grad_weight} | "
        f"type_grad_weight={args.type_grad_weight} | "
        f"guide_mode={args.guide_mode}"
    )

    # ── Trainer — identical callback stack to MolJO's own script ──────────────
    trainer = pl.Trainer(
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=cfg.train.ckpt_freq,
        devices=1,
        num_sanity_val_steps=0,
        inference_mode=False,   # required for guidance gradients
        callbacks=[
            NormalizerCallback(normalizer_dict=cfg.data.normalizer_dict),
            CondMolGenValidationCallback(
                dataset=None,
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
            MolVisualizationCallback(
                atom_decoder=cfg.data.atom_decoder,
                colors_dic=cfg.data.colors_dic,
                radius_dic=cfg.data.radius_dic,
            ),
            ReconValidationCallback(val_freq=cfg.train.val_freq),
            DockingTestCallback(
                dataset=None,
                atom_decoder=cfg.data.atom_decoder,
                atom_enc_mode=cfg.data.transform.ligand_atom_mode,
                atom_type_one_hot=False,
                single_bond=True,
                docking_config=cfg.evaluation.docking_config,
            ),
        ],
    )

    trainer.test(model, dataloaders=test_loader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
