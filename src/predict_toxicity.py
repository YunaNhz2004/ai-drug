from rdkit import Chem
import torch
# pip install torch-geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer
from model import MultimodalNet
from features import molecule_to_graph
from data_utils import label_cols_ordered
from smiles_to_2d import save_smiles_2d

def predict_single_smiles(smiles, model, tokenizer, device):  
    model.eval()
    # --- build graph ---
    mol = Chem.MolFromSmiles(smiles)
    g = molecule_to_graph(mol)      
    g = g.to(device)

    # --- tokenize ---
    encoded = tokenizer(
        smiles,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # --- prepare batch dict giống training ---
    batch = {
        'graph': g,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    with torch.no_grad():
        outputs = model(batch)

        bin_prob = torch.sigmoid(outputs['binary_logits']).cpu().numpy()[0, 0]
        organ_probs = torch.sigmoid(outputs['organ_logits']).cpu().numpy()[0]
        adr_probs = torch.sigmoid(outputs['adr_logits']).cpu().numpy()[0]

    return bin_prob, organ_probs, adr_probs


def print_prediction_with_labels(bin_p, organ_p, adr_p, label_cols):
    print("\n===== FULL PREDICTION =====\n")

    # Binary toxicity
    print(f"{label_cols[0]}: {bin_p:.4f}")

    print("\n--- ORGAN TOXICITY (19 labels) ---")
    for label, prob in zip(label_cols[1:20], organ_p):
        print(f"{label}: {prob:.4f}")

    print("\n--- ADR (8 labels) ---")
    for label, prob in zip(label_cols[20:], adr_p):
        print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    CHECKPOINT = './checkpoints/trained_multimodal_net.pth'


    # smiles_test = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    smiles_test = "[O-][N+](=O)C1=CC=C(Cl)C=C1"


    device = torch.device("cpu")
    # Atom feat dim 
    atom_feat_dim = 41
    tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

    # --- REBUILD MODEL GIỐNG LÚC TRAIN ---
    model = MultimodalNet(atom_feat_dim=atom_feat_dim,
                        text_model_name='seyonec/ChemBERTa-zinc-base-v1',
                        graph_emb_dim=128, text_emb_dim=128, fusion_dim=128,
                        num_binary=1, num_organs=19, num_adr=8)

        # --- LOAD WEIGHTS ---
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()

    bin_p, organ_p, adr_p = predict_single_smiles(
        smiles_test, model, tokenizer, device
    )

    print_prediction_with_labels(bin_p, organ_p, adr_p, label_cols_ordered)

    save_smiles_2d(smiles_test)
    
