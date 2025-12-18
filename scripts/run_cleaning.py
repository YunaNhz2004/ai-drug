import sys
import os
import pandas as pd
from tqdm import tqdm

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.append(project_root)

# Định nghĩa thư mục
DATA_DIR = os.path.join(project_root, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')             
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed') 

# Tạo folder nếu chưa có
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Import hàm clean
try:
    from src.data_utils import clean_smiles
except ImportError:
    print("❌ Lỗi: Không tìm thấy module 'src'.")
    sys.exit(1)

# --- 2. HÀM XỬ LÝ VÀ LƯU FILE PICKLE (.PKL) ---
def process_and_save_df(filename):
    input_path = os.path.join(RAW_DIR, filename)
    
    # ĐỔI ĐUÔI FILE: .csv -> .pkl
    output_filename = filename.replace('.csv', '.pkl')
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    
    print(f"\n🔄 Đang xử lý: {filename}...")
    
    if not os.path.exists(input_path):
        print(f"⚠️  BỎ QUA: Không tìm thấy file tại '{input_path}'")
        return

    # Đọc CSV gốc
    df = pd.read_csv(input_path)
    
    # Tìm cột SMILES
    smiles_col = None
    if 'smiles' in df.columns: smiles_col = 'smiles'
    elif 'SMILES' in df.columns: smiles_col = 'SMILES'
    
    if smiles_col:
        # Làm sạch dữ liệu
        tqdm.pandas(desc="   Cleaning")
        df['cleaned_smiles'] = df[smiles_col].progress_apply(clean_smiles)
        
        # Loại bỏ dòng lỗi
        df_clean = df.dropna(subset=['cleaned_smiles'])
        
        # Reset index cho đẹp (quan trọng khi lưu dạng df)
        df_clean = df_clean.reset_index(drop=True)
        
        # --- LƯU DẠNG DATAFRAME (.pkl) ---
        df_clean.to_pickle(output_path)
        
        print(f"✅ Đã lưu file DataFrame: {output_filename}")
        print(f"   - Đường dẫn: {output_path}")
        print(f"   - Kích thước: {len(df_clean)} dòng")
    else:
        print("❌ Lỗi: Không tìm thấy cột 'smiles' trong file.")

# --- 3. CHẠY ---
if __name__ == "__main__":
    print(f"📂 Input:  {RAW_DIR}")
    print(f"📂 Output: {PROCESSED_DIR} (Format: .pkl)")
    print("-" * 50)
    
    files = ['tox21.csv', 'toxcast.csv', 'sider.csv']
    
    for f in files:
        process_and_save_df(f)
        
    print("-" * 50)
    print("🎉 XONG! Bây giờ bạn có thể dùng pd.read_pickle() để đọc file.")