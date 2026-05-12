"""
Water-mediated pharmacophore and metal coordination guidance for MolJO.

Guidance functions are drop-in replacements for the ClassifierScoreModel
used in MolJO's sampling loop.  Each class exposes:

    .input_type  ("parameter")
    .prop_name   (set externally by configure_classifiers)
    .interdependency_modeling(time, protein_pos, protein_v,
                               batch_protein, batch_ligand,
                               theta_h_t, mu_pos_t, gamma_coord)
       -> (exp_pred [B], atom_prop [N_ligand, 1])

Gradients from exp_pred.log() back-propagate through mu_pos_t and
theta_h_t via standard PyTorch autograd.
"""

from .water_pharmacophore import WaterPharmacophoreGuidance
from .zinc_coordination import ZincCoordinationGuidance
from .utils import (
    parse_waters_from_pdb,
    parse_metal_from_pdb,
    compute_pocket_centroid,
    transform_to_model_space,
)

__all__ = [
    "WaterPharmacophoreGuidance",
    "ZincCoordinationGuidance",
    "parse_waters_from_pdb",
    "parse_metal_from_pdb",
    "compute_pocket_centroid",
    "transform_to_model_space",
]
