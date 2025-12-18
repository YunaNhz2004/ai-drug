
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
# pip install torch torchvision torchaudio

# pip install torch-geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random

def smiles_to_molecule(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False) #Lọc molecule bị valence lỗi
        Chem.SanitizeMol(molecule)
        return molecule
    except:
        return None
    

# ============================ get_atom_features =======================
def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == s) for s in permitted_list]
    return binary_encoding



def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_atoms = ['C','N','O','S','F','Si','P','Cl','Br','I','B','Na','K','Ca','Fe','Zn','Cu','Unknown']
    if not hydrogens_implicit:
        permitted_atoms = ['H'] + permitted_atoms
    
    atom_type_enc = one_hot_encoding(atom.GetSymbol(), permitted_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(min(atom.GetDegree(), 4), [0,1,2,3,4])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-1,0,1])
    hybridisation_enc = one_hot_encoding(str(atom.GetHybridization()), ["S","SP","SP2","SP3"])
    is_in_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    # optional numeric features
    atomic_mass_scaled = [atom.GetMass() / 100]
    
    atom_feature_vector = (
        atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc +
        hybridisation_enc + is_in_ring_enc + is_aromatic_enc + atomic_mass_scaled
    )

    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                         ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"])
        atom_feature_vector += chirality_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(min(atom.GetTotalNumHs(), 4), [0,1,2,3,4])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


# test 
mol = Chem.MolFromSmiles("CCO")

for atom in mol.GetAtoms():
    print("Atom:", atom.GetSymbol())
    print(get_atom_features(atom))
    print("Length:", len(get_atom_features(atom)))
    print("-"*40)

# ====================================================================== 

def get_bond_features(bond, use_stereochemistry=True):
    bond_type_enc = one_hot_encoding(
        bond.GetBondType(),
        [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    )
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_enc = one_hot_encoding(
            str(bond.GetStereo()),
            ["STEREOZ","STEREOE","STEREONONE"]
        )
        bond_feature_vector += stereo_enc

    return np.array(bond_feature_vector)


def molecule_to_graph(molecule,
                      use_atom_features=True, use_bond_features=True,
                      hydrogens_implicit=True):
    """
    Convert RDKit Mol -> PyG Data
    """
    if molecule is None:
        return None

    # ===== NODE FEATURES (Atom) =====
    atom_features = []
    for atom in molecule.GetAtoms():
        feat = get_atom_features(atom, use_chirality=True, hydrogens_implicit=hydrogens_implicit)
        atom_features.append(feat)

    x = torch.tensor(np.vstack(atom_features), dtype=torch.float) if len(atom_features) > 0 else torch.zeros((0,1), dtype=torch.float)

    # ===== EDGE FEATURES (Bond) =====
    edge_index = []
    edge_attr_list = []

    for bond in molecule.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # Lấy feature cho cạnh
        bf = get_bond_features(bond) if use_bond_features else np.array([1.0])

        # Thêm cạnh 2 chiều (Undirected graph)
        edge_index.append([u, v])
        edge_attr_list.append(bf)

        edge_index.append([v, u])
        edge_attr_list.append(bf)

    # Chuyển đổi sang Tensor
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        # Tính kích thước feature ảo để tránh lỗi shape
        dummy_bond = get_bond_features(Chem.MolFromSmiles("CC").GetBondBetweenAtoms(0,1))
        edge_attr = torch.zeros((0, len(dummy_bond)), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.vstack(edge_attr_list), dtype=torch.float)

    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    return data

