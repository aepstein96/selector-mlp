import pandas as pd
import matplotlib.pyplot as plt

# Create feature selection visualization comparing performance with varying numbers of features
def plotFeatureSelectionResults(results_df):
    results_plot = results_df.plot()
    results_plot.set_xlabel('Number of features selected')
    results_plot.set_ylabel('Accuracy') 
    results_plot.set_ylim(0, 1)
    return results_plot

# Generate accuracy and loss plots from training metrics
def plotTrainingResults(metrics_df):
    # Filter to only include evaluation rows (where train_accuracy is not null)
    eval_df = metrics_df.dropna(subset=['train_accuracy'])
    
    # Create accuracy plot
    accuracy_cols = ['train_accuracy', 'val_accuracy', 'train_multiclass_accuracy', 'val_multiclass_accuracy']
    accuracy_plot = eval_df.plot(x='epoch', y=accuracy_cols, figsize=(10, 6))
    accuracy_plot.set_xlabel('Epoch')
    accuracy_plot.set_ylabel('Accuracy')
    accuracy_plot.set_ylim(0, 1)
    accuracy_plot.set_title('Training and Validation Accuracy')
    
    # Create loss plot
    loss_cols = ['train_eval_loss', 'val_eval_loss']
    loss_plot = eval_df.plot(x='epoch', y=loss_cols, figsize=(10, 6))
    loss_plot.set_xlabel('Epoch')
    loss_plot.set_ylabel('Loss')
    loss_plot.set_title('Training and Validation Loss')
    
    return accuracy_plot, loss_plot

    
