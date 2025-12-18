# =========================
# FILE: src/data_utils.py (CHỈ THAY PHẦN LOAD DATA)
# =========================
import pandas as pd
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import SaltRemover

# Giữ nguyên path cũ
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
DATA_DIR = os.path.join(project_root, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


# =========================
# UTILS (GIỮ NGUYÊN)
# =========================
def clean_smiles(smiles_raw: str):
    try:
        mol = Chem.MolFromSmiles(smiles_raw)
        if mol is None:
            return None
        mol = SaltRemover.SaltRemover().StripMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


# =========================
# LOAD DATASET (MỚI - THAY THẾ PHẦN LOAD PKL CŨ)
# =========================
def load_dataset_from_csv(dataset_name):
    """Load từ CSV thay vì pickle cũ"""
    csv_path = os.path.join(RAW_DIR, f'{dataset_name}.csv')
    
    print(f"[INFO] Loading {dataset_name.upper()} from CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Cannot load {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Tìm cột SMILES
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    
    # Clean SMILES
    print(f"[INFO] Cleaning SMILES for {dataset_name}...")
    df['cleaned_smiles'] = df[smiles_col].apply(clean_smiles)
    df = df.dropna(subset=['cleaned_smiles']).reset_index(drop=True)
    
    # Add source flag
    df['sources'] = [[dataset_name]] * len(df)
    
    print(f"[OK] {dataset_name} loaded: {df.shape}")
    return df


# =========================
# LOAD 3 DATASETS (THAY THẾ PHẦN CŨ)
# =========================
print("[INFO] Loading datasets from CSV...")

tox21_df = load_dataset_from_csv('tox21')
sider_df = load_dataset_from_csv('sider')
toxcast_df = load_dataset_from_csv('toxcast')


# =========================
# MERGE DATASETS (GIỮ NGUYÊN)
# =========================
def build_merged_dataframe(tox21_df, sider_df, toxcast_df):
    tox21, sider, toxcast = tox21_df.copy(), sider_df.copy(), toxcast_df.copy()

    for df in [tox21, sider, toxcast]:
        if "cleaned_smiles" not in df.columns:
            df["cleaned_smiles"] = df["smiles"]

    tox21["mol_id"] = [f"TOX21_{i+1}" for i in range(len(tox21))]
    sider["mol_id"] = [f"SIDER_{i+1}" for i in range(len(sider))]
    toxcast["mol_id"] = [f"TOXCAST_{i+1}" for i in range(len(toxcast))]

    all_smiles = pd.concat(
        [
            tox21[["mol_id", "cleaned_smiles"]],
            sider[["mol_id", "cleaned_smiles"]],
            toxcast[["mol_id", "cleaned_smiles"]],
        ],
        ignore_index=True,
    ).drop_duplicates("cleaned_smiles")

    merged = all_smiles.rename(columns={"mol_id": "any_mol_id"})

    def prefix(df, p):
        excl = ["mol_id", "smiles", "cleaned_smiles", "id"]
        labels = [c for c in df.columns if c not in excl and c != "sources"]
        df = df.rename(columns={c: f"{p}_{c}" for c in labels})
        df = df.rename(columns={"sources": f"{p}_sources"})
        return df

    merged = merged.merge(prefix(tox21, "tox21"), on="cleaned_smiles", how="left")
    merged = merged.merge(prefix(sider, "sider"), on="cleaned_smiles", how="left")
    merged = merged.merge(prefix(toxcast, "toxcast"), on="cleaned_smiles", how="left")

    def merge_sources(row):
        s = []
        for p in ["tox21", "sider", "toxcast"]:
            col = f"{p}_sources"
            if isinstance(row.get(col), list):
                s.extend(row[col])
        return sorted(set(s))

    merged["sources"] = merged.apply(merge_sources, axis=1)

    print("[MERGED]", merged.shape)
    return merged


# =========================
# SELECT LABELS (GIỮ NGUYÊN)
# =========================
def select_label_columns(merged_df):
    cols = merged_df.columns.tolist()
    toxcast_cols = [c for c in cols if c.startswith("toxcast_")]

    merged_df["any_toxicity"] = (
        merged_df[toxcast_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        .gt(0)
        .astype(float)
    )

    sider_cols = [c for c in cols if c.startswith("sider_")]
    organ_cols = [c for c in sider_cols if any(k in c.lower() for k in ["liver","kidney","renal","hepatic"])]
    adr_cols = [c for c in sider_cols if c not in organ_cols]

    return ["any_toxicity"] + organ_cols + adr_cols, organ_cols, adr_cols


# =========================
# RUN (GIỮ NGUYÊN BIẾN GLOBAL)
# =========================
merged_df = build_merged_dataframe(tox21_df, sider_df, toxcast_df)
label_cols_ordered, organ_cols, adr_cols = select_label_columns(merged_df)

print("Final labels:", len(label_cols_ordered))
print(label_cols_ordered[:20])


# =========================
# EXPORT (ĐỂ INFERENCE IMPORT ĐƯỢC)
# =========================
__all__ = [
    'clean_smiles',
    'merged_df', 
    'label_cols_ordered',
    'organ_cols',
    'adr_cols'
]