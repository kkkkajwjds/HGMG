import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaGraphFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.h = nn.Parameter(torch.ones(embed_dim, 1))

    def forward(self, H_list):
        meta_reps = H_list
        meta_scores = []
        for X_mg in meta_reps:
            pooled = X_mg.mean(dim=0, keepdim=True)  # [1, d]
            e_m = torch.matmul(pooled, self.h) / X_mg.shape[1]  # [1,1]
            meta_scores.append(e_m)
        meta_scores = torch.stack(meta_scores, dim=0)  # [M,1]
        alpha = F.softmax(meta_scores, dim=0)  # [M,1]
        X_mg_final = sum(alpha[i] * meta_reps[i] for i in range(len(meta_reps)))
        return X_mg_final, alpha