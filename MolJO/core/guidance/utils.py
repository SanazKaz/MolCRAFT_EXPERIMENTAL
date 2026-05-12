"""
Utilities for extracting pharmacophore data from crystal structures and
transforming coordinates into MolJO's internal model space.

MolJO's model space applies two transforms to raw crystal coordinates:
  1. Divide by pos_normalizer  (a scalar, default 1.0)
  2. Subtract protein pocket centroid  (mean of pocket heavy-atom positions)

Both transforms are invertible; this module handles the forward direction.
"""

import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# PDB parsing helpers
# ---------------------------------------------------------------------------

def parse_waters_from_pdb(
    pdb_path: str,
    chain: Optional[str] = None,
    max_bfactor: float = 999.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract crystallographic water oxygen positions and their B-factors.

    Waters are identified as HETATM records with residue name HOH or WAT.
    Only waters on the specified chain (or all chains if chain is None) are
    returned.  Waters with B-factor > max_bfactor are excluded.

    Args:
        pdb_path:   Path to the PDB file.
        chain:      Single-character chain identifier, or None for all chains.
        max_bfactor: Exclude waters with B-factor above this threshold.

    Returns:
        positions  – float32 array [N_water, 3] in Angstrom.
        bfactors   – float32 array [N_water] (crystallographic B-factors).
    """
    positions = []
    bfactors = []

    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            res_name = line[17:20].strip()
            if res_name not in ("HOH", "WAT", "H2O"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "O":  # oxygen only
                continue
            rec_chain = line[21]
            if chain is not None and rec_chain != chain:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66])
            except ValueError:
                continue
            if b > max_bfactor:
                continue
            positions.append([x, y, z])
            bfactors.append(b)

    if not positions:
        raise ValueError(
            f"No waters found in {pdb_path} "
            f"(chain={chain}, max_bfactor={max_bfactor})"
        )
    return np.array(positions, dtype=np.float32), np.array(bfactors, dtype=np.float32)


def parse_metal_from_pdb(
    pdb_path: str,
    element: str = "ZN",
    chain: Optional[str] = None,
) -> np.ndarray:
    """
    Return position(s) of a metal ion from a PDB file.

    Searches both ATOM and HETATM records for the given element symbol.

    Args:
        pdb_path: Path to the PDB file.
        element:  Element symbol, e.g. "ZN", "MG", "CA" (case-insensitive).
        chain:    Chain identifier, or None for all chains.

    Returns:
        positions – float32 array [N_metal, 3] in Angstrom.
    """
    positions = []
    element_upper = element.upper().strip()

    with open(pdb_path, "r") as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            atom_name = line[12:16].strip().upper()
            elem_field = line[76:78].strip().upper() if len(line) > 76 else ""
            # match by element column or atom name
            if elem_field != element_upper and atom_name != element_upper:
                continue
            rec_chain = line[21]
            if chain is not None and rec_chain != chain:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            positions.append([x, y, z])

    if not positions:
        raise ValueError(f"No {element} found in {pdb_path}")
    return np.array(positions, dtype=np.float32)


def parse_pocket_protein_atoms(pdb_path: str, is_pocket: bool = True) -> np.ndarray:
    """
    Return heavy-atom positions for all protein ATOM records.

    Args:
        pdb_path: Path to a (pocket) PDB file.
        is_pocket: If True, skip residue-level filtering (file is already a pocket).

    Returns:
        positions – float32 array [N_atoms, 3] in Angstrom.
    """
    positions = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):  # skip hydrogens
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            positions.append([x, y, z])
    if not positions:
        raise ValueError(f"No ATOM records found in {pdb_path}")
    return np.array(positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Coordinate transformation helpers
# ---------------------------------------------------------------------------

def compute_pocket_centroid(pocket_atom_positions: np.ndarray) -> np.ndarray:
    """
    Compute the mean position of pocket atoms (used for centering in MolJO).

    This mirrors the `center_pos(mode='protein')` transform applied during
    data loading and validation callbacks.

    Args:
        pocket_atom_positions: float array [N, 3].

    Returns:
        centroid – float32 array [3].
    """
    return pocket_atom_positions.mean(axis=0).astype(np.float32)


def transform_to_model_space(
    positions: np.ndarray,
    pocket_centroid: np.ndarray,
    pos_normalizer: float = 1.0,
) -> np.ndarray:
    """
    Convert raw crystal coordinates to MolJO model space.

    Model space = (crystal_coords - pocket_centroid) / pos_normalizer

    Args:
        positions:       float array [N, 3] in Angstrom.
        pocket_centroid: float array [3] – mean of pocket heavy-atom positions.
        pos_normalizer:  scalar divider (cfg.data.normalizer_dict.pos).

    Returns:
        transformed – float32 array [N, 3] in model space.
    """
    return ((positions - pocket_centroid) / pos_normalizer).astype(np.float32)


def bfactor_to_weight(
    bfactors: np.ndarray,
    temperature: float = 10.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert B-factors to pharmacophore point weights.

    Lower B-factor → more conserved water → higher weight.
    Weight = softmax(-B / temperature).

    Args:
        bfactors:    float array [N].
        temperature: scaling factor for the softmax.
        normalize:   if True, weights sum to 1.

    Returns:
        weights – float32 array [N].
    """
    neg_b = -bfactors / temperature
    neg_b -= neg_b.max()  # numerical stability
    weights = np.exp(neg_b)
    if normalize:
        weights = weights / (weights.sum() + 1e-8)
    return weights.astype(np.float32)


def filter_waters_near_pocket(
    water_positions: np.ndarray,
    bfactors: np.ndarray,
    pocket_centroid: np.ndarray,
    radius: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only waters within `radius` Angstrom of the pocket centroid.

    Args:
        water_positions: float array [N, 3].
        bfactors:        float array [N].
        pocket_centroid: float array [3].
        radius:          distance cutoff in Angstrom.

    Returns:
        (filtered_positions [M, 3], filtered_bfactors [M])
    """
    dists = np.linalg.norm(water_positions - pocket_centroid, axis=1)
    mask = dists <= radius
    if mask.sum() == 0:
        raise ValueError(
            f"No waters found within {radius} Å of pocket centroid. "
            "Try increasing the radius."
        )
    return water_positions[mask], bfactors[mask]
