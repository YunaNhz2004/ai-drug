import os
import io
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import urllib.parse
from PIL import Image

def encode(smile):
    return urllib.parse.quote(smile, safe="")

def decode(filename):
    return urllib.parse.unquote(filename)

# def show_image(filepath):
#     if not os.path.exists(filepath):
#         print(f"File không tồn tại: {filepath}")
#         return
    
#     try:
#         # Đọc ảnh
#         img = mpimg.imread(filepath)
        
#         # Hiển thị ảnh
#         plt.imshow(img)
#         plt.axis('off')  # tắt trục
#         plt.show()
        
#     except Exception as e:
#         print(f"Lỗi khi hiển thị ảnh: {e}")

import base64
from io import BytesIO

def smiles_to_3d_molblock(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(
        mol,
        AllChem.ETKDGv3()
    )

    AllChem.UFFOptimizeMolecule(mol)

    mol = Chem.RemoveHs(mol)

    return Chem.MolToMolBlock(mol)
def smiles_2d_to_base64(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # hoặc raise Exception

    img = Draw.MolToImage(mol, size=(300, 300))
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def smiles_3d_to_base64(smiles: str) -> str:
    """
    Chuyển SMILES → 3D structure → Base64 (molblock format)
    Frontend sẽ dùng 3Dmol.js để render
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol)
        
        mol = Chem.RemoveHs(mol)
        
        # Convert to MOL format
        molblock = Chem.MolToMolBlock(mol)
        
        # Encode to base64
        return base64.b64encode(molblock.encode()).decode()
    
    except Exception as e:
        print(f"Lỗi tạo 3D structure: {e}")
        return ""


def save_smiles_2d(smiles, save_dir="outputs/molecules/2d"):

    # 2. File sẽ dùng SMILES encode làm tên
    filename = encode(smiles)
    filepath = os.path.join(save_dir, f"{filename}.png")

    # 3. Nếu file đã tồn tại → in ảnh ra luôn
    if os.path.exists(filepath):
        print(f"Đã tồn tại: {filepath}")
        # show_image(filepath)
        return filepath

    # 4. Chuyển SMILES → Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES không hợp lệ: {smiles}")
    
    # 5. Generate toạ độ 2D
    Chem.rdDepictor.Compute2DCoords(mol)

    # Drawing options
    opts = rdMolDraw2D.MolDrawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True
    opts.explicitMethyl = True
    opts.useBWAtomPalette = True

    # 6. Vẽ ảnh
    # img = Draw.MolToImage(mol, size=(400, 400))
    # Render
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    drawer.SetDrawOptions(opts)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    # 7. Lưu
    # Convert bytes → PIL image
    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    img.save(filepath)
    print(f"Đã lưu: {filepath}")

    # 8. Hiển thị ảnh
    # show_image(filepath)

    return filepath
