import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import SaltRemover
import pandas as pd
import numpy as np

def clean_smiles(smiles_raw: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles_raw)
        if mol is None: return None
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None


TOX21 = './data/processed/tox21.pkl'
SIDER = './data/processed/sider.pkl'
TOXCAST = './data/processed/toxcast.pkl'


tox21_df = pd.read_pickle(TOX21)
sider_df = pd.read_pickle(SIDER)
toxcast_df = pd.read_pickle(TOXCAST)

tox21_df['sources'] = [['tox21']] * len(tox21_df)
sider_df['sources'] = [['sider']] * len(sider_df)
toxcast_df['sources'] = [['toxcast']] * len(toxcast_df)
#test
print('tox21_df: ',type(tox21_df))
print(tox21_df.head())
print('===================================')
print('sider_df: ',type(sider_df))
print(sider_df.head())
print('===================================')
print('toxcast_df: ',type(toxcast_df))
print(toxcast_df.head())
print('===================================')


def build_merged_dataframe(tox21_df, sider_df, toxcast_df, smiles_col='cleaned_smiles'):
    """
    Hợp nhất 3 nguồn dữ liệu (Tox21, SIDER, ToxCast) dựa trên cleaned_smiles.
    Tự động prefix tên cột label để tránh trùng lặp và đảm bảo traceability.
    """

    # ---- Copy kèm flag nguồn ----
    tox21 = tox21_df.copy()
    sider = sider_df.copy()
    toxcast = toxcast_df.copy()

    # ---- Đảm bảo có cleaned_smiles ----
    for df in [tox21, sider, toxcast]:
        if "cleaned_smiles" not in df.columns:
            if "smiles" in df.columns:
                print('DataFrame ko có cleaned_smiles ==> dùng tạm cột smiles')
                df["cleaned_smiles"] = df["smiles"]
            else:
                raise ValueError("DataFrame thiếu cả smiles và cleaned_smiles")

    # -----------------------------
    # Tạo mol_id chuẩn hoá
    # -----------------------------

    # Tox21: giữ nguyên (nếu không có thì tạo)
    if "mol_id" not in tox21.columns:
        tox21["mol_id"] = ["TOX21_" + str(i+1) for i in range(len(tox21))]

    # SIDER: tạo SIDER_1..n
    sider["mol_id"] = ["SIDER_" + str(i+1) for i in range(len(sider))]

    # ToxCast: tạo TOXCAST_1..n
    toxcast["mol_id"] = ["TOXCAST_" + str(i+1) for i in range(len(toxcast))]

    # ---- Tạo danh sách smile duy nhất ----
    all_smiles = pd.concat(
        [
            tox21[["mol_id", "cleaned_smiles"]],
            sider[["mol_id", "cleaned_smiles"]],
            toxcast[["mol_id", "cleaned_smiles"]],
        ],
        ignore_index=True
    ).drop_duplicates(subset=["cleaned_smiles"]).reset_index(drop=True) # XÓA HÀNG TRÙNG SMILES 

    all_smiles = all_smiles.drop_duplicates(subset=["cleaned_smiles"]).reset_index(drop=True)

    # DataFrame nền tảng chứa unified SMILES
    merged = all_smiles[["mol_id", "cleaned_smiles"]].rename(columns={"mol_id": "any_mol_id"})

    # ---- Hàm prefix cột ----
    def prefix_labels(src_df, prefix):
        df = src_df.copy()
        exclude_cols = ["mol_id", "smiles", "cleaned_smiles", "id"]

        label_cols = [c for c in df.columns if c not in exclude_cols]

        # Giữ lại sources  không prefix
        for keep in ["sources"]:
            if keep in label_cols:
                label_cols.remove(keep)
        # Prefix các cột label thật sự     
        df = df.rename(columns={c: f"{prefix}_{c}" for c in label_cols})

        # Rename sources để merge không bị đè
        if "sources" in df.columns:
            df = df.rename(columns={"sources": f"{prefix}_sources"})
        
        return df

    tox21_pref = prefix_labels(tox21, "tox21")
    sider_pref = prefix_labels(sider, "sider")
    toxcast_pref = prefix_labels(toxcast, "toxcast")

    # ---- 5. Merge left theo cleaned_smiles ----
    merged = merged.merge(tox21_pref, on="cleaned_smiles", how="left")
    merged = merged.merge(sider_pref, on="cleaned_smiles", how="left")
    merged = merged.merge(toxcast_pref, on="cleaned_smiles", how="left")

    # Gom sources từ 3 dataset

    def merge_sources(row):
        s = []
        for pref in ["tox21", "sider", "toxcast"]:
            col = f"{pref}_sources"
            if col in row and isinstance(row[col], list):
                s.extend(row[col])
        return list(sorted(set(s)))

    merged["sources"] = merged.apply(merge_sources, axis=1)


    print("===== MERGED DATAFRAME =====")
    print("Shape:", merged.shape)
    print("Preview (transpose):")
    print(merged.head(5))

    return merged

def select_label_columns(merged_df):
    cols = merged_df.columns.tolist()

    # Detect prefix dạng 1 underscore
    # - lấy label ko lấy sources 
    tox21_cols = [c for c in cols if c.startswith("tox21_") and "sources" not in c]
    sider_cols = [c for c in cols if c.startswith("sider_") and "sources" not in c]
    toxcast_cols = [c for c in cols if c.startswith("toxcast_") and "sources" not in c]


    print("Detected tox21 cols:", tox21_cols[:10])
    print("Detected sider cols:", sider_cols[:10])
    print("Detected toxcast cols:", toxcast_cols[:10])

    # ---- Binary head ----
    # any label toxcast = 1 thì sample toxic
    def compute_is_toxic(row):
        if len(toxcast_cols) == 0:
            return np.nan
        # vals = row[tox21_cols].astype(float).fillna(0).values
        # - đảm bảo là number 0/1 
        vals = pd.to_numeric(row[toxcast_cols], errors="coerce").fillna(0).values
        return float(vals.sum() > 0)

    # Sinh nhãn binary thật
    merged_df["any_toxicity"] = merged_df.apply(compute_is_toxic, axis=1)

    # ---- Organ head: chọn SIDER columns có 'disorder' ----

    # mapping các keyword liên quan Organ / System trong SIDER
    organ_keywords = [
        "disorder",         # original logic
        "system",           # ví dụ: nervous_system_disorder
        "cardio", "renal", "hepatic", "liver", "kidney",
        "immune", "respiratory", "pulmonary"
    ]


    # tách sider thành 2 nhóm organ + ADR 
    # + CHỌN CỘT NÀO CHỨA TỪ KHÓA ORGAN 
    organ_cols = [
        c for c in sider_cols
        if any(kw in c.lower() for kw in organ_keywords)
    ]

    # + FALLBACK

    if len(organ_cols) == 0:
        print('CỘT ORGAN ĐANG RỖNG ==> LẤY TẠM FALLBACK')
        binary_sider = [
            c for c in sider_cols
            if merged_df[c].dropna().isin([0,1]).all()
        ]
        organ_cols = binary_sider[:5]    # lấy 5 cột đầu tiên thay vì 3 bừa
        print("!!! Fallback: SIDER không có 'organ keywords', lấy 5 binary columns ĐỂ THỬ NGHIỆM.")

    
    # ---- ADR head ----
    # + CHỈ GIỮ CÁC CỘT NHỊ PHÂN
    adr_cols = [c for c in sider_cols if c not in organ_cols]

    # chỉ giữ ADR là cột nhị phân 0/1
    adr_cols = [
        c for c in adr_cols
        if merged_df[c].dropna().isin([0,1]).all()
    ]


    if len(adr_cols) == 0:
        adr_cols = toxcast_cols[:10]  # fallback
        print("!!! Fallback ADR → lấy 10 cột toxcast đầu tiên.")

    print("-> Binary:", "any_toxicity") # in log 
    print("-> Organ cols ({}): {}".format(len(organ_cols), organ_cols))
    print("-> ADR cols ({}): {}".format(len(adr_cols), adr_cols[:20]))

    label_cols_ordered = ["any_toxicity"] + organ_cols + adr_cols
    return label_cols_ordered, organ_cols, adr_cols



# test
merged_df = build_merged_dataframe(tox21_df, sider_df, toxcast_df)
label_cols_ordered, organ_cols, adr_cols = select_label_columns(merged_df)

print("Final label order length:", len(label_cols_ordered))
print("First labels:", label_cols_ordered[:20])


