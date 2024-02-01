import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.CCE import CCE

class CSG2A_net(nn.Module):
    def __init__(self, gex_dim, hdim, ppi_adj, dropout=0.1, bias=False):
        super(CSG2A_net, self).__init__()
        self.CCE = CCE(gex_dim)
        self.gex_dim = gex_dim
        self.hdim = hdim
        self.ppi_adj = ppi_adj # (gex_dim, gex_dim)
        self.w_gex = nn.Parameter(torch.randn(gex_dim, hdim))
        self.w_comp = nn.Parameter(torch.randn(gex_dim, hdim))
        
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(gex_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gex_dim, gex_dim, bias=bias)
        )

    def forward(self, b_gex, node_feat, mask, adj_matrix, dist_matrix, dose, time):
        comp = self.CCE(node_feat, mask, adj_matrix, dist_matrix, dose, time) # (batch, gex_dim)
        q_gex = b_gex.unsqueeze(-1) * self.w_gex # (batch, gex_dim) -> (batch, gex_dim, hdim)
        q_comp = comp.unsqueeze(-1) * self.w_comp # (batch, gex_dim) -> (batch, gex_dim, hdim)
        q_added = q_gex + q_comp

        CSG2A_score = torch.matmul(q_added, q_added.transpose(1, 2)) / np.sqrt(self.hdim) # (batch, gex_dim, gex_dim)
        CSG2A_weight = CSG2A_score + self.ppi_adj 

        pred_gex = torch.einsum('ij,ijk->ij', b_gex, CSG2A_weight)
        pred_gex = self.feed_forward(pred_gex)

        return pred_gex, comp
    
class CSG2A_finetune(nn.Module):
    def __init__(self, gex_dim, CSG2A_params, finetune_hdim1 = 512, finetune_hdim2 = 64, 
                 dropout = 0.1):
        super(CSG2A_finetune, self).__init__()
        self.CSG2A = CSG2A_net(**CSG2A_params)

        # scaling layer
        self.scalar = nn.Parameter(torch.abs(torch.randn(1))+1e-8) #prevent zeroing out
        self.bias = nn.Parameter(torch.randn(1))

        # prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(finetune_hdim1*3,finetune_hdim1),
            nn.BatchNorm1d(finetune_hdim1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(finetune_hdim1,finetune_hdim2),
            nn.BatchNorm1d(finetune_hdim2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(finetune_hdim2,1)
            )
        
        # projection layers
        self.agex_proj=nn.Sequential(
            nn.Linear(gex_dim,finetune_hdim1),
            nn.BatchNorm1d(finetune_hdim1),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.bgex_proj=nn.Sequential(
            nn.Linear(gex_dim,finetune_hdim1),
            nn.BatchNorm1d(finetune_hdim1),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.comp_proj=nn.Sequential(
            nn.Linear(gex_dim,finetune_hdim1),
            nn.BatchNorm1d(finetune_hdim1),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def scaling_layer(self, gex):
        return (gex + self.bias) * self.scalar

    def forward(self, b_gex, node_feat, mask, adj_matrix, dist_matrix, dose, time):
        scaled_gex = self.scaling_layer(b_gex)
        pred_gex, comp = self.CSG2A(scaled_gex, node_feat, mask, adj_matrix, dist_matrix, dose, time)

        output = torch.cat([self.bgex_proj(b_gex), 
                       self.agex_proj(pred_gex), 
                       self.comp_proj(comp)], axis=1)        
        output = self.pred_head(output)
        return output, comp