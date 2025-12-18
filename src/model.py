import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=128, heads=4, dropout=0.1, use_gatv2=False, debug=False):
        super().__init__()
        Conv = GATv2Conv if use_gatv2 else GATConv

        # conv1 produces hidden_dim (concat heads)
        self.conv1 = Conv(in_dim, hidden_dim // heads, heads=heads, concat=True)
        # conv2 produces out_dim (concat heads)
        self.conv2 = Conv(hidden_dim, out_dim // heads, heads=heads, concat=True)

        self.out_dim = self.conv2.out_channels  # real output feature dim after conv2
        self.act = nn.ReLU()
        self.pool = global_mean_pool
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x, edge_index, edge_attr, batch_idx):
        # device-safe: determine device from batch_idx if x may be None
        device = batch_idx.device if isinstance(batch_idx, torch.Tensor) else (x.device if isinstance(x, torch.Tensor) else torch.device("cpu"))

        # empty graph handling
        if x is None or x.shape[0] == 0:
            B = (batch_idx.max().item() + 1) if isinstance(batch_idx, torch.Tensor) and batch_idx.numel() > 0 else 1
            if self.debug:
                print(f"GATEncoder DEBUG: empty graph, returning zeros {(B, self.out_dim)} on {device}")
            return torch.zeros((B, self.out_dim), device=device, dtype=torch.float)

        # standard forward (note: GATConv/GATv2Conv ignore edge_attr unless using edge_dim)
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.act(x)

        x = self.pool(x, batch_idx)  # [B, out_dim]
        if self.debug:
            print("GATEncoder DEBUG: pooled node_emb shape:", x.shape)
        return x


class TextEncoder(nn.Module):
    def __init__(self, hf_model_name='seyonec/ChemBERTa-zinc-base-v1', proj_dim=128, freeze_backbone=False, debug=False):
        super().__init__()
        self.hf = AutoModel.from_pretrained(hf_model_name)
        self.proj = nn.Linear(self.hf.config.hidden_size, proj_dim)
        self.act = nn.ReLU()
        self.debug = debug

        if freeze_backbone:
            for p in self.hf.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.hf(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Roberta-style models: CLS token is at index 0
        cls = out.last_hidden_state[:, 0, :]   # [B, hidden_size]
        cls = self.act(self.proj(cls))         # [B, proj_dim]
        if self.debug:
            print("TextEncoder DEBUG: cls shape:", cls.shape)
        return cls


class ModalityAttentionFusion(nn.Module):
    def __init__(self, input_dims, common_dim=128):
        super().__init__()
        self.n = len(input_dims)
        self.projs = nn.ModuleList([nn.Linear(d, common_dim) for d in input_dims])
        self.attn_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(common_dim, 64), nn.Tanh(), nn.Linear(64, 1))
            for _ in range(self.n)
        ])
        self.final_proj = nn.Linear(common_dim, common_dim)

    def forward(self, embeddings):
        # embeddings: list of [B, d_i]
        projeds = [F.relu(p(e)) for p, e in zip(self.projs, embeddings)]       # list of [B, common_dim]
        scores = [fc(p).squeeze(-1) for fc, p in zip(self.attn_fc, projeds)]    # each [B]
        scores = torch.stack(scores, dim=1)    # [B, n]
        weights = torch.softmax(scores, dim=1) # [B, n]
        stacked = torch.stack(projeds, dim=1) # [B, n, common_dim]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1) # [B, common_dim]
        fused = F.relu(self.final_proj(fused))
        return fused, weights


class MultimodalNet(nn.Module):
    def __init__(self, atom_feat_dim, text_model_name, graph_emb_dim=128, text_emb_dim=128, fusion_dim=128,
                 num_binary=1, num_organs=1, num_adr=1, debug=False):
        super().__init__()
        self.graph_enc = GATEncoder(in_dim=atom_feat_dim, out_dim=graph_emb_dim, debug=debug)
        self.text_enc = TextEncoder(hf_model_name=text_model_name, proj_dim=text_emb_dim, debug=debug)
        self.fusion = ModalityAttentionFusion([graph_emb_dim, text_emb_dim], common_dim=fusion_dim)
        self.binary_head = nn.Linear(fusion_dim, num_binary)
        self.organ_head = nn.Linear(fusion_dim, num_organs)
        self.adr_head = nn.Linear(fusion_dim, num_adr)
        self.debug = debug

    def forward(self, batch):
        g = batch['graph']
        node_x = getattr(g, 'x', None)
        edge_index = getattr(g, 'edge_index', None)
        edge_attr = getattr(g, 'edge_attr', None)   # optional
        batch_idx = getattr(g, 'batch', None)

        if self.debug:
            print("Model forward DEBUG: node_x shape:", None if node_x is None else tuple(node_x.shape))

        graph_emb = self.graph_enc(node_x, edge_index, edge_attr, batch_idx)   # [B, graph_emb_dim]
        text_emb = self.text_enc(batch['input_ids'], batch['attention_mask'])  # [B, text_emb_dim]

        fused, weights = self.fusion([graph_emb, text_emb])                    # [B, fusion_dim], [B, 2]
        bin_logits = self.binary_head(fused)
        organ_logits = self.organ_head(fused)
        adr_logits = self.adr_head(fused)

        if self.debug:
            print("Model outputs shapes -> bin:", bin_logits.shape, "organ:", organ_logits.shape, "adr:", adr_logits.shape)
        return {'binary_logits': bin_logits, 'organ_logits': organ_logits, 'adr_logits': adr_logits, 'fusion_weights': weights}
