"""
Zinc coordination geometry guidance for MolJO BFN sampling.

This module scores generated molecules on how well they present atoms capable
of coordinating a zinc ion in a tetrahedral geometry.  The score is fully
differentiable with respect to both mu_pos_t (atom positions) and theta_h_t
(atom type probabilities), so gradients flow through both channels in MolJO's
BFN parameter space.

Biology background
------------------
In carbonic anhydrase II (CA-II) the catalytic zinc is coordinated by three
histidine residues from the protein and one solvent/inhibitor atom.  Typical
Zn–N/O/S bond lengths are 2.0–2.3 Å.  The geometry around zinc is close to
tetrahedral (ideal angle: 109.5°).

Score design
------------
For each ligand atom j in molecule b we compute three terms:

  1. **Distance score**  d_j  = exp(-( ||mu_j - z|| - d_target )^2 / r_sigma^2)
     Peaks when the atom is at the target Zn–X distance (default 2.15 Å).

  2. **Type score**      t_j  = sum of theta_h_t[j, k] for k in coord_type_indices
     This is the probability that atom j is a coordinating type (N, O, S).
     Differentiable w.r.t. theta_h_t.

  3. **Combined atom score**   a_j = d_j * t_j   ∈ [0, 1]

The per-molecule expected coordination number is:
     E_coord_b = sum_j a_j  (soft count of coordinating atoms)

A target coordination number of n_target is rewarded via:
     coord_score_b = exp(-(E_coord_b - n_target)^2 / n_sigma^2)

An additional tetrahedral angle penalty is applied to atom pairs that are
both within the coordination shell:
     For pairs (j, k) with a_j > angle_threshold and a_k > angle_threshold:
         cos_theta = dot((mu_j - z)/d_j, (mu_k - z)/d_k)
         ang_pen = (cos_theta - cos_ideal)^2   [ideal ≈ -1/3 for tetrahedral]
     angle_score_b = exp(-mean(ang_pen) / ang_sigma^2)

Final per-molecule score:
     exp_pred_b = sigmoid(
         w_coord * log(coord_score_b + eps)
         + w_angle * log(angle_score_b + eps)
         - offset
     )

Coordinate convention
---------------------
zinc_pos must be provided in MolJO model space:
    model_pos = (crystal_pos - pocket_centroid) / pos_normalizer

Atom type indices (add_aromatic, 13 classes)
--------------------------------------------
    0: H    1: C    2: C_ar  3: N    4: N_ar
    5: O    6: O_ar 7: F     8: P    9: P_ar
    10: S   11: S_ar 12: Cl

Coordinating types by default: N (3,4), O (5,6), S (10,11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from typing import List, Optional
import numpy as np


# Default coordinating atom type indices for 'add_aromatic' 13-class encoding.
# N (aromatic + non-aromatic), O (aromatic + non-aromatic), S (aromatic + non-aromatic)
DEFAULT_COORD_TYPE_INDICES: List[int] = [3, 4, 5, 6, 10, 11]

# Cosine of 109.47° (tetrahedral ideal angle)
COS_TETRAHEDRAL = -1.0 / 3.0


class ZincCoordinationGuidance(nn.Module):
    """
    Differentiable zinc-coordination guidance for MolJO sampling.

    Args:
        zinc_pos:              Tensor or ndarray [3] – zinc position, model space.
        d_target:              Target Zn–X bond distance in model-space units.
        r_sigma:               Width of the distance Gaussian (model units).
        n_target:              Target number of coordinating ligand atoms (float).
        n_sigma:               Width of the coordination-number Gaussian.
        coord_type_indices:    Atom type indices considered coordinating.
        w_coord:               Weight of the coordination-number term in log-space.
        w_angle:               Weight of the angle term in log-space.
        ang_sigma:             Width of the angle deviation penalty (radians).
        angle_threshold:       Min combined atom score to include atom in angle calc.
        offset:                Scalar offset before final sigmoid (tunes sensitivity).
        device:                Torch device string.
    """

    def __init__(
        self,
        zinc_pos,
        d_target: float = 2.15,
        r_sigma: float = 0.3,
        n_target: float = 1.5,
        n_sigma: float = 0.8,
        coord_type_indices: Optional[List[int]] = None,
        w_coord: float = 1.0,
        w_angle: float = 0.5,
        ang_sigma: float = 0.3,
        angle_threshold: float = 0.05,
        offset: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()

        if not isinstance(zinc_pos, torch.Tensor):
            zinc_pos = torch.tensor(np.asarray(zinc_pos, dtype=np.float32))

        self.register_buffer("zinc_pos", zinc_pos.float().reshape(3))

        if coord_type_indices is None:
            coord_type_indices = DEFAULT_COORD_TYPE_INDICES
        self.coord_type_indices = coord_type_indices

        self.d_target = d_target
        self.r_sigma = r_sigma
        self.n_target = n_target
        self.n_sigma = n_sigma
        self.w_coord = w_coord
        self.w_angle = w_angle
        self.ang_sigma = ang_sigma
        self.angle_threshold = angle_threshold
        self.offset = offset

        # MolJO interface attributes
        self.input_type = "parameter"
        self.prop_name = "zinc_coordination"

        self.to(device)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _score_batch(
        self,
        mu_pos_t: torch.Tensor,   # [N_ligand, 3] requires_grad=True
        theta_h_t: torch.Tensor,  # [N_ligand, K] requires_grad=True
        batch_ligand: torch.Tensor,  # [N_ligand] long
    ) -> tuple:
        """
        Compute per-molecule coordination scores.

        Returns:
            exp_pred  – [B] scores in (0, 1), differentiable.
            atom_prop – [N_ligand, 1] per-atom coordination scores.
        """
        zinc = self.zinc_pos.to(mu_pos_t.device)
        n_mols = int(batch_ligand.max().item()) + 1

        # 1. Distance score: exp(-(d - d_target)^2 / r_sigma^2)
        disp = mu_pos_t - zinc.unsqueeze(0)           # [N, 3]
        dist = torch.norm(disp, dim=-1, keepdim=False) + 1e-8   # [N]
        dist_score = torch.exp(
            -((dist - self.d_target) ** 2) / (self.r_sigma ** 2)
        )   # [N]

        # 2. Type score: probability of being a coordinating atom type
        type_indices = torch.tensor(
            self.coord_type_indices, dtype=torch.long, device=mu_pos_t.device
        )
        type_score = theta_h_t[:, type_indices].sum(dim=-1).clamp(0.0, 1.0)   # [N]

        # 3. Combined per-atom coordination score
        atom_score = dist_score * type_score   # [N]
        atom_prop = atom_score.unsqueeze(-1)   # [N, 1]

        # 4. Coordination number score (per molecule)
        E_coord = scatter_sum(atom_score, batch_ligand, dim=0, dim_size=n_mols)  # [B]
        coord_score = torch.exp(
            -((E_coord - self.n_target) ** 2) / (self.n_sigma ** 2)
        )   # [B]

        # 5. Tetrahedral angle score
        # For atoms with high combined score, penalise angle deviations.
        # This term is computed per pair (j, k) within the same molecule.
        angle_score = self._angle_score(disp, dist, atom_score, batch_ligand, n_mols)

        # 6. Final per-molecule score (log-space combination → sigmoid)
        eps = 1e-8
        log_score = (
            self.w_coord * torch.log(coord_score + eps)
            + self.w_angle * torch.log(angle_score + eps)
        )
        exp_pred = torch.sigmoid(log_score - self.offset)   # [B]

        return exp_pred, atom_prop

    def _angle_score(
        self,
        disp: torch.Tensor,      # [N, 3] displacement from zinc to atom
        dist: torch.Tensor,      # [N] distances
        atom_score: torch.Tensor,# [N] per-atom coordination score
        batch_ligand: torch.Tensor,
        n_mols: int,
    ) -> torch.Tensor:
        """
        Compute a soft tetrahedral angle penalty per molecule.

        Only atom pairs with atom_score > angle_threshold contribute.
        The ideal tetrahedral cosine is -1/3 (≈ 109.47°).

        Uses a soft weighting: each pair (j,k) is weighted by a_j * a_k,
        so the penalty naturally fades for atoms unlikely to be coordinating.

        Returns:
            angle_score – [B] in (0, 1].
        """
        device = disp.device
        n_mols_val = n_mols

        angle_score = torch.ones(n_mols_val, device=device, dtype=disp.dtype)

        # Unit vectors from zinc toward each atom
        unit = disp / dist.unsqueeze(-1)   # [N, 3]

        # Compute per-molecule angle scores iteratively to support variable
        # atom counts while keeping the computation differentiable.
        for mol_idx in range(n_mols_val):
            mask = batch_ligand == mol_idx
            if mask.sum() < 2:
                continue

            u_mol = unit[mask]      # [M, 3]
            a_mol = atom_score[mask]  # [M]

            # Pairwise weights: a_j * a_k
            pair_w = a_mol.unsqueeze(0) * a_mol.unsqueeze(1)  # [M, M]

            # Pairwise cosines via dot product of unit vectors
            cos_mat = torch.mm(u_mol, u_mol.t())   # [M, M]

            # Angle penalty: (cos - cos_ideal)^2
            pen_mat = (cos_mat - COS_TETRAHEDRAL) ** 2  # [M, M]

            # Mask diagonal (self-pairs) – weight is still a_j^2 but angle=0
            diag_mask = torch.eye(mask.sum(), device=device, dtype=torch.bool)
            pair_w = pair_w.masked_fill(diag_mask, 0.0)

            w_sum = pair_w.sum() + 1e-8
            weighted_pen = (pair_w * pen_mat).sum() / w_sum   # scalar

            angle_score[mol_idx] = torch.exp(-weighted_pen / (self.ang_sigma ** 2))

        return angle_score   # [B]

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
        Compute zinc-coordination guidance score.

        Both mu_pos_t (positions) and theta_h_t (type probabilities) enter
        the computation, so both pos_grad and type_grad are non-zero.

        Args:
            time:         [N_ligand, 1] (unused).
            protein_pos:  [N_protein, 3] (unused).
            protein_v:    [N_protein, feat_dim] (unused).
            batch_protein:[N_protein] (unused).
            batch_ligand: [N_ligand] molecule indices.
            theta_h_t:    [N_ligand, K] atom-type parameters, requires_grad=True.
            mu_pos_t:     [N_ligand, 3] coordinate parameters, requires_grad=True.
            gamma_coord:  [N_ligand, 1] (unused).
            return_all:   legacy (unused).
            fix_x:        legacy (unused).

        Returns:
            exp_pred  – [B] in (0, 1), differentiable w.r.t. mu_pos_t and theta_h_t.
            atom_prop – [N_ligand, 1] per-atom scores.
        """
        self.zinc_pos = self.zinc_pos.to(mu_pos_t.device)
        return self._score_batch(mu_pos_t, theta_h_t, batch_ligand)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str,
        pocket_centroid,
        pos_normalizer: float = 1.0,
        element: str = "ZN",
        chain: Optional[str] = None,
        which: int = 0,
        **kwargs,
    ) -> "ZincCoordinationGuidance":
        """
        Build guidance from a PDB file containing a metal ion.

        Args:
            pdb_path:        PDB file path.
            pocket_centroid: [3] array – centroid of pocket atoms (crystal coords).
            pos_normalizer:  Scalar – cfg.data.normalizer_dict.pos.
            element:         Metal element symbol (e.g. "ZN", "MG").
            chain:           PDB chain (None = all).
            which:           Index into the list of found metal positions (0 = first).
            **kwargs:        Forwarded to ZincCoordinationGuidance.__init__.

        Returns:
            ZincCoordinationGuidance instance.
        """
        from .utils import parse_metal_from_pdb, transform_to_model_space

        metal_positions = parse_metal_from_pdb(pdb_path, element=element, chain=chain)
        if which >= len(metal_positions):
            raise ValueError(
                f"Requested index {which} but only {len(metal_positions)} {element} "
                f"atoms found in {pdb_path}"
            )
        crystal_pos = metal_positions[which]
        model_pos = transform_to_model_space(crystal_pos.reshape(1, 3), pocket_centroid, pos_normalizer)[0]

        # Convert d_target and r_sigma from Angstrom to model space if normalizer != 1
        if "d_target" not in kwargs:
            kwargs["d_target"] = 2.15 / pos_normalizer
        if "r_sigma" not in kwargs:
            kwargs["r_sigma"] = 0.3 / pos_normalizer

        print(
            f"[ZincCoordinationGuidance] {element} at model-space position "
            f"{model_pos.tolist()} "
            f"(crystal: {crystal_pos.tolist()})"
        )

        return cls(zinc_pos=model_pos, **kwargs)
