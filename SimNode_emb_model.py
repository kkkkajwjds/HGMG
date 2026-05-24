import torch
import torch.nn as nn
import torch.nn.functional as F

def build_similarity_adj(X, k=10):
    X_norm = F.normalize(X, p=2, dim=1)
    sim = torch.mm(X_norm, X_norm.t())  # cosine similarity

    N = X.shape[0]
    A = torch.zeros((N, N), device=X.device)

    for i in range(N):
        topk = torch.topk(sim[i], k + 1).indices  # include self
        for j in topk:
            if i != j:
                A[i, j] = 1.0
    return A

class SimGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, X, A):

        # layer 1
        X1 = A @ X
        X1 = self.W1(X1)
        X1 = F.leaky_relu(X1)

        # layer data
        X2 = A @ X1
        X2 = self.W2(X2)
        X2 = F.leaky_relu(X2)

        return X2

class FinalFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(2 * d, d)

    def forward(self, X_mg, X_sim):

        if X_sim.shape[0] != X_mg.shape[0]:
            X_sim = X_sim[:X_mg.shape[0]]

        X = torch.cat([X_mg, X_sim], dim=-1)
        return self.fc(X)


class FullModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.sim_gnn = SimGNN(
            in_dim=feature_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim
        )

        self.fusion = FinalFusion(hidden_dim)

    def forward(self, X_u, X_mg, k3=50):

        A_sim = build_similarity_adj(X_u, k3)

        X_sim = self.sim_gnn(X_u, A_sim)

        F = self.fusion(X_mg, X_sim)

        return F

