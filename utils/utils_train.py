import torch
import numpy as np
from scipy import stats

def train(model, trainloader, optimizer, criterion, epoch, args, logger=print):
    model.train()
    train_losses = []
    for idx, batch in enumerate(trainloader):
        adj_mat, node_feat, dist_mat, b_gex, y_target, dose, time = batch

        adj_mat = adj_mat.to(args.device) # (batch_size, Max_comp_len, Max_comp_len)
        node_feat = node_feat.to(args.device) # (batch_size, Max_comp_len, node_feature)
        dist_mat = dist_mat.to(args.device) # (batch_size, Max_comp_len, Max_comp_len)
        b_gex = b_gex.to(args.device) # (batch_size, gene_dim)

        if y_target.dim() == 1: # if label is scalar: unsqueeze for loss calculation
            y_target = y_target.unsqueeze(-1)
        y_target = y_target.to(args.device) # (batch_size, gene_dim)
        dose = dose.unsqueeze(-1).to(args.device)
        time = time.unsqueeze(-1).to(args.device)

        SMILES_mask = torch.sum(torch.abs(node_feat), dim=-1) != 0 # 64x67 (batch_size, Max_comp_len)

        y_pred, _ = model(b_gex, node_feat, SMILES_mask, adj_mat, dist_mat,dose,time)

        optimizer.zero_grad()
        train_loss = criterion(y_pred, y_target)
        train_loss.backward()
        optimizer.step()
        
        if epoch==1:
            if idx % 100 ==0:
                logger(f'- batch{idx+1}/{len(trainloader)} of epoch{epoch}, loss: {train_loss}')

        train_losses.append(train_loss.item())
    mean_train_loss = np.mean(train_losses)
    
    return mean_train_loss

def eval(model, loader, criterion, args):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            adj_mat, node_feat, dist_mat, b_gex, y_target, dose, time = batch

            adj_mat = adj_mat.to(args.device) # (batch_size, Max_comp_len, Max_comp_len)
            node_feat = node_feat.to(args.device) # (batch_size, Max_comp_len, node_feature)
            dist_mat = dist_mat.to(args.device) # (batch_size, Max_comp_len, Max_comp_len)
            b_gex = b_gex.to(args.device) # (batch_size, gene_dim)
            
            if y_target.dim() == 1: # if label is scalar: unsqueeze for loss calculation
                y_target = y_target.unsqueeze(-1)
            y_target = y_target.to(args.device) # (batch_size, gene_dim)
            dose = dose.unsqueeze(-1).to(args.device)
            time = time.unsqueeze(-1).to(args.device)            

            SMILES_mask = torch.sum(torch.abs(node_feat), dim=-1) != 0 # 64x67 (batch_size, Max_comp_len)

            y_pred, _ = model(b_gex, node_feat, SMILES_mask, adj_mat, dist_mat,dose,time)
            
            preds.append(y_pred)
            targets.append(y_target)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    valid_loss = criterion(preds, targets)
    valid_pcc = stats.pearsonr(preds.cpu().numpy().reshape(-1), targets.cpu().numpy().reshape(-1))[0]
    return valid_loss.item(), valid_pcc
