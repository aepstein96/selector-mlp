from torch import nn
import torch
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import numpy as np

# PyTorch Lightning wrapper for SelectorMLP model
# In theory extensible to other deep multiclass classifiers, but not currently implemented
# Handles training, validation, logging, and optimization
class MultiClassifier(pl.LightningModule):

    def __init__(self, Net,  num_features, num_classes, lossFunc=nn.CrossEntropyLoss(), lr=0.001, alpha=0.01, noise_std=0, optimizer=optim.AdamW, betas=(0.9, 0.999), weight_decay=0.01, **net_args):
        super().__init__()
        self.save_hyperparameters(ignore=['Model', 'lossFunc', 'optimizer'])
        self.net = Net(num_features, num_classes, **net_args)
        self.optimizer = optimizer
        self.lossFunc = lossFunc
        self.accFunc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='micro')
        self.balanced_accFunc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        
        # Initialize lists to collect predictions and targets
        self.val_step_outputs = []
        self.val_step_targets = []
        self.train_step_outputs = []
        self.train_step_targets = []
        
    # Save model to file
    def save(self, path):
        torch.save(self.net, path)
        
    def forward(self, data): # Final predictions, including softmax
        return torch.softmax(self.net(data), dim=1)

    # Single training step with optional regularization
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.net(X + self.hparams.noise_std*torch.randn(X.shape, device=self.device))
        penalty_term = self.net.regularize()
        criterion = nn.CrossEntropyLoss()
        loss_term = criterion(y_pred, y)
        loss = loss_term + self.hparams.alpha * penalty_term # Custom regularization term from net
        self.log("train_batch_loss", loss, on_step=True, on_epoch=False)
        self.log('penalty_term', penalty_term, on_step=True, on_epoch=False)
        self.log("train_batch_loss_term", loss_term, on_step=True, on_epoch=False)
        return loss

    # Validation step - handles both validation and training evaluation
    # Dataloader idx 0: validation. Dataloader idx 1: training (optional)
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        y_pred = self.net(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y)
        
        if dataloader_idx == 0: # Validation
            self.log('val_eval_loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.val_step_outputs.append(y_pred.detach())
            self.val_step_targets.append(y.detach())
        elif dataloader_idx == 1: # Training
            self.log('train_eval_loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.train_step_outputs.append(y_pred.detach())
            self.train_step_targets.append(y.detach())
        
        return {"loss": loss, "preds": y_pred, "targets": y}
    
    def on_validation_epoch_end(self):
        # Calculate accuracy metrics on all validation data at once
        if self.val_step_outputs:
            # Ensure all tensors are on the same device (GPU)
            device = self.device
            all_val_preds = torch.cat(self.val_step_outputs).to(device)
            all_val_targets = torch.cat(self.val_step_targets).to(device)
            
            # Create fresh metric instances for calculating on all data
            accFunc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes, average='micro').to(device)
            balancedAccFunc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes, average='macro').to(device)
            
            val_accuracy = accFunc(all_val_preds, all_val_targets)
            val_multiclass_accuracy = balancedAccFunc(all_val_preds, all_val_targets)
            
            self.log('val_accuracy', val_accuracy)
            self.log('val_multiclass_accuracy', val_multiclass_accuracy)
            
            # Clear the lists for next epoch
            self.val_step_outputs.clear()
            self.val_step_targets.clear()
        
        # Do the same for training data if available
        if self.train_step_outputs:
            # Ensure all tensors are on the same device (GPU)
            device = self.device
            all_train_preds = torch.cat(self.train_step_outputs).to(device)
            all_train_targets = torch.cat(self.train_step_targets).to(device)
            
            accFunc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes, average='micro').to(device)
            balancedAccFunc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes, average='macro').to(device)
            
            train_accuracy = accFunc(all_train_preds, all_train_targets)
            train_multiclass_accuracy = balancedAccFunc(all_train_preds, all_train_targets)
            
            self.log('train_accuracy', train_accuracy)
            self.log('train_multiclass_accuracy', train_multiclass_accuracy)
            
            # Clear the lists for next epoch
            self.train_step_outputs.clear()
            self.train_step_targets.clear()
            
    # Configure optimizer with hyperparameters
    def configure_optimizers(self):
        return self.optimizer(self.net.parameters(), lr=self.hparams.lr, betas=self.hparams.betas, weight_decay=self.hparams.weight_decay)

    
# Feature selector layer that applies weights to input features
# Modified torch.nn.Linear, borrowed from PyTorch github (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py)
# Has no bias, and weights are put through a sigmoid so the result is between 0 and 1
# If clamp is True, weights are clamped between 0 and 1
# Features is the number of features in the input data
class Selector(nn.Module):
    def __init__(self, num_features, std=None, clamp=False):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.empty(num_features))
        self.register_parameter('bias', None)
        self.clamp = clamp
        self.std = std
        if std is None:
            if clamp:
                self.std = 0.2 # default std for clamped selector
            else:
                self.std = 0.2 # default std for regular selector
            
        self.reset_parameters()

    # Initialize weights with normal distribution
    def reset_parameters(self, std=None):
        if std is None:
            std = self.std
            
        nn.init.normal_(self.weight, mean=0, std=std)
        
        # Apply absolute value to weights during initialization if clamped
        if self.clamp:
            self.weight.data = torch.abs(self.weight.data)

    # Apply feature weights to input via element-wise multiplication
    def forward(self, x):
        if self.clamp:
            self.weight.data = torch.clamp(self.weight.data, min=0, max=1)
        return torch.mul(x, self.weight) # Element-wise multiplication


# MLP with optional feature selection layer
# Implements a three-layer network with configurable regularization
class SelectorMLP(nn.Module):
    def __init__(self, num_features, num_classes, selector_type=None, penalty="L1", batch_norm="true", L1_size=512, L2_size=128, L3_size=64, leak_angle=0.2, dropout=0.3):
        super().__init__()

        # Initialize selector layer based on specified type
        if selector_type is None or selector_type == "none":
            self.selector = None
        elif selector_type == 'clamped':
            self.selector = Selector(num_features, clamp=True)
        elif selector_type == 'std':
            self.selector = Selector(num_features, clamp=False)
        else:
            raise ValueError(f"Invalid selector type: {selector_type}")

        self.penalty = penalty

        # Configure batch normalization layers
        if batch_norm == "true":
            self.batch1 = nn.BatchNorm1d(L1_size)
            self.batch2 = nn.BatchNorm1d(L2_size)
            self.batch3 = nn.BatchNorm1d(L3_size)
        else:
            self.batch1 = nn.Identity()
            self.batch2 = nn.Identity()
            self.batch3 = nn.Identity()

        # Define network layers
        self.h1 = nn.Linear(num_features, L1_size)
        self.h2 = nn.Linear(L1_size, L2_size)
        self.h3 = nn.Linear(L2_size, L3_size)

        self.out = nn.Linear(L3_size, num_classes)
        self.relu = nn.LeakyReLU(leak_angle)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.selector is not None:
            x = self.selector(x)
            
        x = self.dropout(self.relu(self.batch1(self.h1(x))))
        x = self.dropout(self.relu(self.batch2(self.h2(x))))
        x = self.dropout(self.relu(self.batch3(self.h3(x))))
        return self.out(x)

    # Regularize the selector weights
    # Used to minimize the number of features/genes the model depends on
    def regularize(self, penalty=None):
        if penalty is None:
            penalty = self.penalty
        
        if self.selector is None or penalty is None:
            return 0
        elif penalty == 'L1':
            return torch.abs(self.selector.weight).sum()
        elif penalty == 'L2':
            return torch.sqrt((self.selector.weight**2).sum())
        else:
            raise ValueError(f"Invalid regularization type: {penalty}")
