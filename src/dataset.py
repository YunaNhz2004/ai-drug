import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch_geometric.data import Data as GraphData
import features as features


class SmartDrugDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, smiles_col='clean_smiles', label_cols=None):
        """
        Args:
            df: DataFrame chứa dữ liệu (Tox21, ToxCast, hoặc Sider).
            tokenizer: HuggingFace Tokenizer.
            max_len: Độ dài tối đa của chuỗi text.
            smiles_col: Tên cột chứa chuỗi SMILES (vd: 'clean_smiles', 'smiles').
            label_cols: Danh sách tên cột nhãn. Nếu None, tự động lấy các cột còn lại.
        """
        self.data = []
        print(f"Processing Dataset with {len(df)} samples...")
        
        # 1. Xác định cột nhãn (loại bỏ cột smiles và các cột id nếu có)
        if label_cols is None:
            exclude_cols = [smiles_col, 'mol_id', 'id', 'smiles','cleaned_smiles']
            label_cols = [c for c in df.columns if c not in exclude_cols]
        
        print(f"Detected {len(label_cols)} tasks/labels.")

        # 2. Duyệt qua từng dòng dữ liệu
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            smi = row[smiles_col]
            
            # --- Bước A: Tạo Labels và Mask (Xử lý NaN) ---
            # Lấy giá trị labels dưới dạng numpy array
            raw_labels = row[label_cols].values.astype(float)
            
            # Tạo mask: 1 nếu có dữ liệu, 0 nếu là NaN
            mask_array = ~np.isnan(raw_labels)
            mask_array = mask_array.astype(float)
            
            # Fill NaN bằng 0 (để không lỗi tensor, mask sẽ lo phần tính loss sau)
            labels_array = np.nan_to_num(raw_labels, nan=0.0)

            # --- Bước B: Xử lý Graph ---
            mol = features.smiles_to_molecule(smi)
            if mol is None:
                continue # Bỏ qua chất lỗi
            
            # Gọi hàm tạo graph (truyền labels và mask đã xử lý)
            graph = features.molecule_to_graph(
                molecule=mol,
                y=labels_array,
                mask=mask_array
            )

            # --- Bước C: Xử lý Text (Tokenizer) ---
            text_enc = tokenizer(
                smi, 
                max_length=max_len, 
                padding='max_length', 
                truncation=True, 
                return_tensors="pt"
            )

            # --- Bước D: Lưu trữ ---
            self.data.append({
                'graph': graph,
                'input_ids': text_enc['input_ids'].squeeze(0),
                'attention_mask': text_enc['attention_mask'].squeeze(0),
                'labels': torch.tensor(labels_array, dtype=torch.float)
            })
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
