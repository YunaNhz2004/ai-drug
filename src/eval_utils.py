import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_roc_auc(y_true, y_scores):
    """
    Tính ROC-AUC cho multi-label classification
    Tự động bỏ qua NaN và các cột chỉ có 1 class
    """
    auc_scores = []
    num_labels = y_true.shape[1]
    
    for i in range(num_labels):
        true_col = y_true[:, i]
        pred_col = y_scores[:, i]
        
        # Lọc bỏ NaN
        mask = ~np.isnan(true_col)
        valid_true = true_col[mask]
        valid_pred = pred_col[mask]
        
        # Chỉ tính AUC nếu có cả class 0 và class 1
        if len(np.unique(valid_true)) == 2:
            try:
                auc = roc_auc_score(valid_true, valid_pred)
                auc_scores.append(auc)
            except ValueError:
                pass
                
    return np.mean(auc_scores) if len(auc_scores) > 0 else 0.0


def evaluate_model(model, dataloader, task_name, device):
    """
    Đánh giá model trên 1 dataset cụ thể
    
    Args:
        model: MultimodalNet
        dataloader: DataLoader của task cần eval (với collate_fn=lambda x: x)
        task_name: 'tox21', 'toxcast', hoặc 'sider'
        device: 'cuda' hoặc 'cpu'
    Returns:
        auc_score: ROC-AUC của task này
    """
    model.eval()
    
    from torch_geometric.data import Batch
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            # === Prepare data - batch_data là LIST of dicts ===
            graph_list = [item['graph'] for item in batch_data]
            graph_batch = Batch.from_data_list(graph_list).to(device)
            
            input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(device)
            labels = torch.stack([item['labels'] for item in batch_data])  # Keep on CPU
            
            # === Forward với task cụ thể ===
            outputs = model(graph_batch, input_ids, attention_mask, task=task_name)
            
            # === Convert logits to probabilities ===
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # === Collect predictions & labels ===
            all_preds.append(probs)
            all_labels.append(labels.numpy())
    
    # === Concatenate all batches ===
    if len(all_preds) == 0:
        return 0.0
    
    y_pred = np.vstack(all_preds)  # [N_samples, N_labels]
    y_true = np.vstack(all_labels)  # [N_samples, N_labels]
    
    # === Calculate AUC ===
    auc = calculate_roc_auc(y_true, y_pred)
    
    return auc