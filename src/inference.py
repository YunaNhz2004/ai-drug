from rdkit import Chem
import torch
from transformers import AutoTokenizer
from src.model import MultimodalNet
from src.features import molecule_to_graph
from src.data_utils import label_cols_ordered
from src.smiles_to_2d import save_smiles_2d
from functools import lru_cache
import logging
from src.smiles_to_2d import save_smiles_2d, smiles_2d_to_base64, smiles_3d_to_base64


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# GLOBAL: load 1 lần
# =========================
DEVICE = torch.device("cpu")
CHECKPOINT = "./checkpoints/trained_multimodal_net.pth"
ATOM_FEAT_DIM = 41
TEXT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

# Load model và tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

model = MultimodalNet(
    atom_feat_dim=ATOM_FEAT_DIM,
    text_model_name=TEXT_MODEL,
    graph_emb_dim=128,
    text_emb_dim=128,
    fusion_dim=128,
    num_binary=1,
    num_organs=19,
    num_adr=8
)

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Tối ưu thêm nếu có GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    model = model.to(DEVICE)
    logger.info("Using GPU for inference")

# =========================
# CACHE CHO SMILES ĐÃ DỰ ĐOÁN
# =========================
@lru_cache(maxsize=1000)
def cached_predict(smiles: str):
    """Cache kết quả cho SMILES đã predict"""
    return _predict_internal(smiles)


# =========================
# MAIN API FUNCTION
# =========================
def predict_smiles(smiles: str, skip_image: bool = True): # mặc định ko pop up ảnh lên => True là skip
    """
    Input:  smiles string, skip_image (optional)
    Output: dict (JSON-ready)
    """
    # Validate SMILES trước
    smiles = smiles.strip()
    if not smiles:
        return {"error": "Empty SMILES string"}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}

    # Sử dụng cache nếu đã predict trước đó
    try:
        result = cached_predict(smiles)
        
        # Chỉ generate image khi cần (tốn thời gian)
        if not skip_image:
            save_smiles_2d(smiles)
        
        return result
    except Exception as e:
        logger.error(f"Prediction error for {smiles}: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}


def _predict_internal(smiles: str):
    """Internal prediction function (cacheable)"""
    mol = Chem.MolFromSmiles(smiles)
    
    # --- graph ---
    graph = molecule_to_graph(mol).to(DEVICE)

    # --- tokenize ---
    encoded = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    batch = {
        "graph": graph,
        "input_ids": encoded["input_ids"].to(DEVICE),
        "attention_mask": encoded["attention_mask"].to(DEVICE),
    }

    # Inference
    with torch.no_grad():
        outputs = model(batch)

        bin_prob = torch.sigmoid(outputs["binary_logits"])[0, 0].item()
        organ_probs = torch.sigmoid(outputs["organ_logits"])[0].tolist()
        adr_probs = torch.sigmoid(outputs["adr_logits"])[0].tolist()

    # --- format JSON ---
    result = {
        "smiles": smiles,
        "binary_toxicity": {
            "label": label_cols_ordered[0],
            "probability": round(bin_prob, 4)
        },
        "organ_toxicity": {
            label: round(p, 4)
            for label, p in zip(label_cols_ordered[1:20], organ_probs)
        },
        "adr": {
            label: round(p, 4)
            for label, p in zip(label_cols_ordered[20:], adr_probs)
        },"images": {
        "2d": smiles_2d_to_base64(smiles),
        "3d": smiles_3d_to_base64(smiles) 
    }
    }

    return result


# =========================
# HEALTH CHECK
# =========================
def health_check():
    """Check if model is loaded correctly"""
    try:
        test_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
        _ = predict_smiles(test_smiles, skip_image=True)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}