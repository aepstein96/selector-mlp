from torch import nn
import torch
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import numpy as np

class MultiClassifier(pl.LightningModule):

  def __init__(self, Net,  num_features, num_classes, lossFunc=nn.CrossEntropyLoss(), lr=0.001, alpha=0.01, noise_std=0, optimizer=optim.AdamW, betas=(0.9, 0.999), weight_decay=0.01, **net_args):
    super().__init__()
    self.save_hyperparameters(ignore=['Model', 'lossFunc', 'optimizer'])
    self.net = Net(num_features, num_classes, **net_args)
    self.optimizer = optimizer
    self.lossFunc = lossFunc
    self.accFunc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='micro')
    self.balanced_accFunc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    
  def save(self, path):
    torch.save(self.net, path)
    
  def forward(self, data): # Final predictions, including softmax
    return torch.softmax(self.net(data), dim=1)

  def training_step(self, batch, batch_idx):
    X, y = batch
    y_pred = self.net(X + self.hparams.noise_std*torch.randn(X.shape, device=self.device))
    reg_term = self.net.regularize()
    loss_term = self.lossFunc(y_pred, y)
    loss = loss_term + self.hparams.alpha * reg_term # Custom regularization term from net
    self.log("train_batch_loss", loss, on_step=True, on_epoch=False)
    self.log('reg_term', reg_term, on_step=True, on_epoch=False)
    self.log("train_batch_loss_term", loss_term, on_step=True, on_epoch=False)
    return loss

  # Dataloader idx 0: validation. Dataloader idx 1: training (optional)
  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    X, y = batch
    y_pred = self.net(X)
    loss = self.lossFunc(y_pred, y)
    acc = self.accFunc(torch.softmax(y_pred, dim=1), y)
    multiclass_acc = self.balanced_accFunc(torch.softmax(y_pred, dim=1), y)
    
    if dataloader_idx == 0:
      self.log('val_eval_loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
      self.log('val_accuracy', acc, on_step=False, on_epoch=True, add_dataloader_idx=False)
      self.log('val_multiclass_accuracy', multiclass_acc, on_step=False, on_epoch=True, add_dataloader_idx=False)
    elif dataloader_idx == 1:
      self.log('train_eval_loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
      self.log('train_accuracy', acc, on_step=False, on_epoch=True, add_dataloader_idx=False)
      self.log('train_multiclass_accuracy', multiclass_acc, on_step=False, on_epoch=True, add_dataloader_idx=False)
      
  def configure_optimizers(self):
    return self.optimizer(self.net.parameters(), lr=self.hparams.lr, betas=self.hparams.betas, weight_decay=self.hparams.weight_decay)

    
# Modified torch.nn.Linear, borrowed from PyTorch github (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py)
# Has no bias, and weights are put through a sigmoid so the result is between 0 and 1
# If clamp is True, weights are clamped between 0 and 1
class Selector(nn.Module):
    def __init__(self, features, std=None, clamp=False):
        super().__init__()
        self.features = features
        self.weight = nn.Parameter(torch.empty(features))
        self.register_parameter('bias', None)
        self.clamp = clamp
        self.std = std
        if clamp and std is None:
          self.std = 0.2 # default std for clamped selector
          
        self.reset_parameters()

    def reset_parameters(self, std):
        if std is None:
          std = self.std
            
        nn.init.normal_(self.weight, mean=0, std=std)

    def forward(self, x):
        if self.clamp:
            self.weight.data = torch.clamp(self.weight.data, min=0, max=1)
        return torch.mul(x, self.weight) # Element-wise multiplication


class SelectorMLP(nn.Module):
    def __init__(self, num_features, num_classes, selector_type=None,  batch_norm="true", reg_type='L1', L1_size=512, L2_size=128, L3_size=64, leak_angle=0.2, dropout=0.3):
      super().__init__()

      if selector_type is None or selector_type == "none":
        self.selector = nn.Identity()
      elif selector_type == 'clamped':
        self.selector = Selector(num_features, clamp=True)
      elif selector_type == 'std':
        self.selector = Selector(num_features, clamp=False)
      else:
        raise ValueError(f"Invalid selector type: {selector_type}")

      self.reg_type = reg_type

      if batch_norm == "true":
        self.batch1 = nn.BatchNorm1d(L1_size)
        self.batch2 = nn.BatchNorm1d(L2_size)
        self.batch3 = nn.BatchNorm1d(L3_size)
      else:
        self.batch1 = nn.Identity()
        self.batch2 = nn.Identity()
        self.batch3 = nn.Identity()

      self.h1 = nn.Linear(num_features, L1_size)
      self.h2 = nn.Linear(L1_size, L2_size)
      self.h3 = nn.Linear(L2_size, L3_size)

      self.out = nn.Linear(L3_size, num_classes)
      self.relu = nn.LeakyReLU(leak_angle)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      x = self.selector(x)
      x = self.dropout(self.relu(self.batch1(self.h1(x))))
      x = self.dropout(self.relu(self.batch2(self.h2(x))))
      x = self.dropout(self.relu(self.batch3(self.h3(x))))
      return self.out(x)

    def regularize(self):
      if self.selector is None or self.reg_type is None:
        return 0
      elif self.reg_type == 'L1':
        return torch.abs(self.selector.weight).sum()
      elif self.reg_type == 'L2':
        return torch.sqrt((self.selector.weight**2).sum())
      else:
        raise ValueError(f"Invalid regularization type: {self.reg_type}")
