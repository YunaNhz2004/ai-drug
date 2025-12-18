import torch
import torch.nn.functional as F
import numpy as np

def masked_bce_loss(pred, target, weight=None):
    """
    Binary Cross Entropy với mask cho missing labels (NaN)
    
    Args:
        pred: [batch, num_labels] - Logits (chưa qua sigmoid)
        target: [batch, num_labels] - Labels (chứa NaN cho labels thiếu)
        weight: [num_labels] - Optional, trọng số cho từng label
    Returns:
        loss: Scalar tensor
    """
    # Mask các label KHÔNG phải NaN
    mask = ~torch.isnan(target)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    
    # Thay NaN = 0 để tránh lỗi (sẽ bị mask anyway)
    target_clean = torch.where(torch.isnan(target), torch.zeros_like(target), target)
    
    # Compute loss element-wise (chưa reduce)
    loss = F.binary_cross_entropy_with_logits(pred, target_clean, reduction='none')
    
    # Apply weight TRƯỚC KHI mask (nếu có)
    if weight is not None:
        weight_expanded = weight.unsqueeze(0).expand_as(loss)
        loss = loss * weight_expanded
    
    # Apply mask và tính mean
    masked_loss = loss[mask]
    
    return masked_loss.mean()


def train_hybrid_step(model, optimizer, loader_tox21, loader_toxcast, loader_sider, 
                      iter_tox21, iter_toxcast, iter_sider, device):
    """
    Training 1 step bằng cách lấy 1 batch từ MỖI dataset
    Train tuần tự 3 tasks, accumulate gradients, rồi update chung
    
    Args:
        model: MultimodalNet
        optimizer: torch.optim
        loader_xxx: DataLoader cho từng dataset
        iter_xxx: Iterator của từng DataLoader
        device: 'cuda' hoặc 'cpu'
    Returns:
        (total_loss, iter_tox21, iter_toxcast, iter_sider): Loss và updated iterators
    """
    from torch_geometric.data import Batch
    
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    tasks_completed = 0
    
    # === TASK 1: TOX21 ===
    try:
        batch_data = next(iter_tox21)
    except StopIteration:
        iter_tox21 = iter(loader_tox21)  # Reset iterator
        batch_data = next(iter_tox21)
    
    # Prepare data - batch_data là LIST of dicts
    graph_list = [item['graph'] for item in batch_data]
    graph_batch = Batch.from_data_list(graph_list).to(device)
    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(device)
    labels = torch.stack([item['labels'] for item in batch_data]).to(device)
    
    # Forward với task='tox21'
    outputs = model(graph_batch, input_ids, attention_mask, task='tox21')
    loss1 = masked_bce_loss(outputs, labels)
    
    # Check nếu loss hợp lệ
    if not torch.isnan(loss1):
        loss1.backward()  # Accumulate gradients
        total_loss += loss1.item()
        tasks_completed += 1
    
    # === TASK 2: TOXCAST ===
    try:
        batch_data = next(iter_toxcast)
    except StopIteration:
        iter_toxcast = iter(loader_toxcast)
        batch_data = next(iter_toxcast)
    
    graph_list = [item['graph'] for item in batch_data]
    graph_batch = Batch.from_data_list(graph_list).to(device)
    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(device)
    labels = torch.stack([item['labels'] for item in batch_data]).to(device)
    
    outputs = model(graph_batch, input_ids, attention_mask, task='toxcast')
    loss2 = masked_bce_loss(outputs, labels)
    
    if not torch.isnan(loss2):
        loss2.backward()  # Accumulate gradients
        total_loss += loss2.item()
        tasks_completed += 1
    
    # === TASK 3: SIDER ===
    try:
        batch_data = next(iter_sider)
    except StopIteration:
        iter_sider = iter(loader_sider)
        batch_data = next(iter_sider)
    
    graph_list = [item['graph'] for item in batch_data]
    graph_batch = Batch.from_data_list(graph_list).to(device)
    input_ids = torch.stack([item['input_ids'] for item in batch_data]).to(device)
    attention_mask = torch.stack([item['attention_mask'] for item in batch_data]).to(device)
    labels = torch.stack([item['labels'] for item in batch_data]).to(device)
    
    outputs = model(graph_batch, input_ids, attention_mask, task='sider')
    loss3 = masked_bce_loss(outputs, labels)
    
    if not torch.isnan(loss3):
        loss3.backward()  # Accumulate gradients
        total_loss += loss3.item()
        tasks_completed += 1
    
    # === UPDATE WEIGHTS (1 lần cho cả 3 tasks) ===
    # Clip gradients để tránh exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    avg_loss = total_loss / tasks_completed if tasks_completed > 0 else 0.0
    return avg_loss, iter_tox21, iter_toxcast, iter_sider

