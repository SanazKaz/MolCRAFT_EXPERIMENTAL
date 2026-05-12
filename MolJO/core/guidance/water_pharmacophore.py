"""
Water-mediated pharmacophore guidance for MolJO BFN sampling.

This module implements a differentiable pharmacophore score derived from
conserved crystallographic water positions.  The score is designed to plug
directly into MolJO's existing classifier-guidance framework – it satisfies
the same interface as ClassifierScoreModel without requiring any trained
neural-network weights.

Coordinate convention
---------------------
All positions passed to __init__ must already be in **MolJO model space**:
    model_pos = (crystal_pos - pocket_centroid) / pos_normalizer
Use core.guidance.utils.transform_to_model_space to pre-compute this.

Guidance interface
------------------
The sampling loop in bfn4sbdd.py expects each classifier to expose:

    classifier.input_type            → "parameter"
    classifier.prop_name             → set externally by configure_classifiers
    classifier.interdependency_modeling(
        time, protein_pos, protein_v,
        batch_protein, batch_ligand,
        theta_h_t, mu_pos_t, gamma_coord
    ) → (exp_pred [B], atom_prop [N_ligand, 1])

Gradient flow
-------------
exp_pred is a sigmoid-transformed sum of soft-nearest-neighbour scores;
because mu_pos_t enters through differentiable distance operations,
torch.autograd.grad produces a valid pos_grad that pulls generated atoms
toward pharmacophore points.  theta_h_t does not enter the score, so
type_grad is zero (the function still accepts it for interface compatibility).
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_max
from typing import Optional


class WaterPharmacophoreGuidance(nn.Module):
    """
    Differentiable pharmacophore guidance derived from conserved waters.

    For each pharmacophore point p_i with weight w_i the score measures how
    close the nearest generated atom is to that point via a smooth Gaussian
    kernel.  Scores are summed over all pharmacophore points and mapped
    through a sigmoid to produce per-molecule values in (0, 1).

    Score for molecule b
    --------------------
        For each pharmacophore point i:
            raw_i_b = logsumexp_j (-d(mu_j, p_i)^2 / sigma^2)
                      where j ranges over atoms of molecule b
            # logsumexp is a smooth-max: high when nearest atom is close
        pharm_score_b = sum_i  w_i * sigmoid(raw_i_b)
        exp_pred_b    = sigmoid(pharm_score_b - offset)

    Args:
        pharmacophore_positions: Tensor [N_pharm, 3] or ndarray, model space.
        weights:                 Tensor [N_pharm] – pharmacophore weights,
                                 e.g. derived from B-factors.  Automatically
                                 normalised to sum to 1.
        sigma:                   Gaussian width in model-space units
                                 (~Angstrom if pos_normalizer=1).
        offset:                  Shift applied before the outer sigmoid so
                                 that a "neutral" score maps to ~0.5.
        device:                  Torch device.
    """

    def __init__(
        self,
        pharmacophore_positions,
        weights,
        sigma: float = 1.5,
        offset: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__()

        import numpy as np

        if not isinstance(pharmacophore_positions, torch.Tensor):
            pharmacophore_positions = torch.tensor(
                np.asarray(pharmacophore_positions, dtype=np.float32)
            )
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(np.asarray(weights, dtype=np.float32))

        # Normalise weights so they sum to 1.
        weights = weights / (weights.sum() + 1e-8)

        self.register_buffer("pharm_pos", pharmacophore_positions.float())  # [N_pharm, 3]
        self.register_buffer("weights", weights.float())                     # [N_pharm]
        self.sigma = sigma
        self.offset = offset

        # Interface attributes expected by MolJO sampling loop.
        self.input_type = "parameter"
        self.prop_name = "water_pharmacophore"  # may be overwritten by configure_classifiers

        self.to(device)

    # ------------------------------------------------------------------
    # Core differentiable scoring
    # ------------------------------------------------------------------

    def _score_batch(
        self,
        mu_pos_t: torch.Tensor,   # [N_ligand, 3]  requires_grad=True
        batch_ligand: torch.Tensor,  # [N_ligand] long
    ) -> tuple:
        """
        Compute per-molecule pharmacophore scores.

        Returns:
            exp_pred  – [B] sigmoid scores in (0, 1), differentiable.
            atom_prop – [N_ligand, 1] per-atom soft-nearest-neighbour scores.
        """
        n_mols = int(batch_ligand.max().item()) + 1
        n_pharm = self.pharm_pos.shape[0]

        # Pairwise squared distances: [N_ligand, N_pharm]
        # mu_pos_t: [N, 3]  pharm_pos: [P, 3]
        diff = mu_pos_t.unsqueeze(1) - self.pharm_pos.unsqueeze(0)  # [N, P, 3]
        d_sq = (diff ** 2).sum(dim=-1)                               # [N, P]

        # Gaussian kernel scores (per atom, per pharmacophore point)
        log_kernel = -d_sq / (self.sigma ** 2)   # [N, P]

        # Per-atom summary: sum over pharmacophore points weighted by w_i
        # atom_pharm_score[j] = sum_i  w_i * exp(-d(j,i)^2/sigma^2)
        kernel = torch.exp(log_kernel)  # [N, P]  – differentiable
        atom_weighted = (kernel * self.weights.unsqueeze(0)).sum(dim=-1)  # [N]
        atom_prop = atom_weighted.unsqueeze(-1)  # [N, 1]

        # Per-molecule, per-pharmacophore smooth max (logsumexp over atoms)
        # raw_i_b = logsumexp_j(-d(j,i)^2 / sigma^2) for atoms j in molecule b
        # Shape trick: we compute for all atoms then scatter-logsumexp per molecule.
        # logsumexp via: max + log(sum(exp(x - max)))
        mol_pharm_score = torch.zeros(
            n_mols, n_pharm, device=mu_pos_t.device, dtype=mu_pos_t.dtype
        )
        for pidx in range(n_pharm):
            log_k_col = log_kernel[:, pidx]  # [N]
            # Numerically stable scatter-logsumexp using torch_scatter primitives:
            # max_per_mol[b] = max_j log_k_col[j] for j in molecule b
            max_per_mol, _ = scatter_max(log_k_col, batch_ligand, dim=0, dim_size=n_mols)
            # clamp so exp doesn't underflow when max is -inf (empty molecule)
            max_per_mol = max_per_mol.clamp(min=-1e9)
            shifted = log_k_col - max_per_mol[batch_ligand]  # [N]
            sum_exp = scatter_sum(torch.exp(shifted), batch_ligand, dim=0, dim_size=n_mols)  # [B]
            mol_pharm_score[:, pidx] = max_per_mol + torch.log(sum_exp + 1e-12)

        # Each pharmacophore contribution: sigmoid(raw_i_b) ∈ (0,1)
        pharm_contrib = torch.sigmoid(mol_pharm_score)   # [B, P]

        # Weighted sum over pharmacophore points → scalar per molecule
        pharm_total = (pharm_contrib * self.weights.unsqueeze(0)).sum(dim=-1)  # [B]

        # Final sigmoid maps to (0, 1) – required by MolJO (calls .log())
        exp_pred = torch.sigmoid(pharm_total - self.offset)  # [B]

        return exp_pred, atom_prop

    # ------------------------------------------------------------------
    # MolJO guidance interface
    # ------------------------------------------------------------------

    def interdependency_modeling(
        self,
        time,
        protein_pos,
        protein_v,
        batch_protein,
        batch_ligand,
        theta_h_t,
        mu_pos_t,
        gamma_coord,
        return_all: bool = False,
        fix_x: bool = False,
    ):
        """
        Compute pharmacophore guidance score.

        Only mu_pos_t enters the computation; theta_h_t is accepted for
        interface compatibility but produces zero gradient.

        Args:
            time:         [N_ligand, 1] time embeddings (unused).
            protein_pos:  [N_protein, 3] (unused, positions pre-transformed).
            protein_v:    [N_protein, feat_dim] (unused).
            batch_protein:[N_protein] molecule indices (unused).
            batch_ligand: [N_ligand] molecule indices.
            theta_h_t:    [N_ligand, K] atom-type parameters.
            mu_pos_t:     [N_ligand, 3] coordinate parameters, requires_grad=True.
            gamma_coord:  [N_ligand, 1] (unused in score computation).
            return_all:   legacy flag (unused).
            fix_x:        legacy flag (unused).

        Returns:
            exp_pred  – Tensor [B] in (0, 1), differentiable w.r.t. mu_pos_t.
            atom_prop – Tensor [N_ligand, 1] per-atom scores.
        """
        # Ensure pharmacophore buffers are on the same device as inputs.
        self.pharm_pos = self.pharm_pos.to(mu_pos_t.device)
        self.weights = self.weights.to(mu_pos_t.device)

        return self._score_batch(mu_pos_t, batch_ligand)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str,
        pocket_centroid,
        pos_normalizer: float = 1.0,
        chain: Optional[str] = None,
        max_bfactor: float = 999.0,
        pocket_radius: float = 12.0,
        sigma: float = 1.5,
        offset: float = 0.5,
        device: str = "cpu",
    ) -> "WaterPharmacophoreGuidance":
        """
        Build guidance from a PDB file, performing all coordinate transforms.

        Args:
            pdb_path:        Path to the PDB file containing HETATM water records.
            pocket_centroid: [3] array – centroid of pocket atoms in crystal coords.
            pos_normalizer:  Scalar – same value as cfg.data.normalizer_dict.pos.
            chain:           PDB chain to restrict water search (None = all).
            max_bfactor:     Ignore waters with B-factor above this value.
            pocket_radius:   Only include waters within this radius (Å) of centroid.
            sigma:           Gaussian width for pharmacophore scoring (model units).
            offset:          Sigmoid shift (see class docstring).
            device:          Torch device.

        Returns:
            WaterPharmacophoreGuidance instance ready for use.
        """
        import numpy as np
        from .utils import (
            parse_waters_from_pdb,
            filter_waters_near_pocket,
            transform_to_model_space,
            bfactor_to_weight,
        )

        positions, bfactors = parse_waters_from_pdb(pdb_path, chain=chain, max_bfactor=max_bfactor)
        positions, bfactors = filter_waters_near_pocket(positions, bfactors, pocket_centroid, radius=pocket_radius)
        model_positions = transform_to_model_space(positions, pocket_centroid, pos_normalizer)
        weights = bfactor_to_weight(bfactors)

        print(
            f"[WaterPharmacophoreGuidance] Loaded {len(model_positions)} water "
            f"pharmacophore points (sigma={sigma}, offset={offset})"
        )

        return cls(
            pharmacophore_positions=model_positions,
            weights=weights,
            sigma=sigma,
            offset=offset,
            device=device,
        )
