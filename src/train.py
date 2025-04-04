
def train(model, train_set, val_set, save_dir, name, batch_size, num_workers, lr, batch_norm, noise, dropout):
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    train_eval_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    MLP_best = MultiClassifier(SelectorMLP, num_features, num_classes, lr=lr, batch_norm=batch_norm, noise_std=noise, Selector=None, weight_decay=0.3, dropout=dropout)
    logger1 = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=name)
    logger2 = pl.loggers.CSVLogger(save_dir=save_dir, name=name)
    trainer = pl.Trainer(max_epochs=50, accelerator='gpu', logger=[logger1, logger2], default_root_dir=save_dir)
    trainer.fit(MLP_best, train_loader, val_loader)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# NOTES:
# Conditions this was ued

# Retraining best MLP
save_dir = 'logs_MLP_conditions'
name = 'MLP_best'
selector_models = {}
dropout=0.3
batch_norm=False
noise=0.2


# Training clamped selector
save_dir = 'logs_selector_conditions'
name = 'Selector_clamped'
num_features = 2000
selector_models = {}
dropout=0.3
batch_norm=False
noise=0.2