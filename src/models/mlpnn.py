import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.plotting import plot_loss, plot_roc, plot_cm

class SLENet(nn.Module):
    def __init__(self, input_dim, net_hidden_structure, dropout_rates, output_dim=1):
        super().__init__()
        net_fcs = []
        net_in_shape = input_dim
        for i, net_n_hidden_nodes in enumerate(net_hidden_structure):
            net_fcs.append(torch.nn.Linear(net_in_shape, net_n_hidden_nodes))
            net_fcs.append(torch.nn.BatchNorm1d(net_n_hidden_nodes))
            net_fcs.append(torch.nn.ReLU6())
            net_fcs.append(torch.nn.Dropout(p=dropout_rates[i]))
            net_in_shape = net_n_hidden_nodes
        net_fcs.append(torch.nn.Linear(net_in_shape, output_dim))
        self.net_fcs = torch.nn.ModuleList(net_fcs)
        
    def forward(self, x):
        for net_fc in self.net_fcs:
            x = net_fc(x)
        return x

from torch.utils.data import Dataset
class SLEDataset(Dataset):
    """Self-defined dataset."""

    def __init__(self, X, y):
        """
        Args:
            X: EHR info (np.array)
            y: Case vs. Control (np.array)
        """
        super().__init__()
        self.X = torch.from_numpy(X).float() if not torch.is_tensor(X) else X.float()
        self.y = torch.LongTensor(y) if not torch.is_tensor(y) else y.long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to the logits to get the probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute the intersection
        intersection = (inputs * targets).sum()
        
        # Compute the Dice coefficient
        dice_coefficient = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # Return the Dice Loss
        return 1 - dice_coefficient

# train function
from utils.utils_funcs import AverageMeter
import tqdm

def train(epoch, model, criterion, optimizer, train_loader, gradient_clip=3, device='cpu', disable=False):
    """
    One epoch training
    """
    
    model.train()
    losses = AverageMeter()

    # Using tqdm to display progress
    progress_bar = tqdm.tqdm(train_loader, total=len(train_loader), disable=disable)
    for batch_idx, (data, target) in enumerate(progress_bar):
        batch_size = data.size(0)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ===================forward=====================
        outputs = model(data)
        loss = criterion(outputs.squeeze(-1), target.float().squeeze())

        # ===================backward=====================
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        # ===================meters=====================
        losses.update(loss.item(), batch_size)
        loss_return = losses.avg
    
    return loss_return

def predict_mlp(model, X_test, le_trans_test, classes, threshold=0.5, device='cpu'):
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(np.array(X_test, dtype='float32')).to(device))
        pred_proba = torch.sigmoid(prediction)
    y_pred_proba = pred_proba.cpu().numpy()
    y_pred = classes[(y_pred_proba > threshold).astype('int')]
    y_test_true = classes[le_trans_test]
    
    accuracy = sum(y_pred.squeeze()==y_test_true)/len(y_pred)
    print('Accuracy', accuracy)
    
    from sklearn.metrics import confusion_matrix
    confusion_mat = confusion_matrix(y_test_true, y_pred)
    plot_cm(confusion_mat, classes)
    
    from sklearn.metrics import roc_auc_score
    
    auc = roc_auc_score(le_trans_test, y_pred_proba)
    # In this study, "Case" 1 and "Control" 0; we need "Case" to be postive class
    plot_roc(1-le_trans_test, 1-y_pred_proba, figsize=(8,8))
    
    return y_pred_proba, y_pred, y_test_true, accuracy, confusion_mat, auc


