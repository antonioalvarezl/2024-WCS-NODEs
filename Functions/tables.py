"""
tables.py - Results Table Generation
-----------------------------------
This module provides functions for generating and exporting result tables.
It helps in summarizing training results and metrics for further analysis.

Workflow position: Evaluation & Reporting - summarizes model performance for analysis
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def dataframe(trainer, path):
    """
    Creates a DataFrame with loss values at specific training epochs and saves it to Excel.
    
    Parameters:
    - trainer (Trainer): The trainer object containing training history
    - path (str): Directory path where to save the Excel file
    
    Returns:
    - df (DataFrame): The created pandas DataFrame
    """
    # Calculate specific epoch checkpoints (1/4, 1/2, 3/4, and full training)
    num_epochs = len(trainer.histories['loss_history'])
    specific_epochs = [num_epochs // 4, num_epochs // 2, 3 * num_epochs // 4, num_epochs]
    
    # Define column names including the best epoch
    columns = [f"Epoch {epoch}" for epoch in specific_epochs] + [f'Best Epoch ({trainer.best_epoch})']
    
    # Create full path for the Excel file
    full_path_excel = os.path.join(path, 'Losses.xlsx')

    # Prepare data for the DataFrame
    data = []
    
    # First row: Loss values at specific epochs and at best epoch
    row = [f'{trainer.histories["loss_history"][epoch - 1]:.5f}' for epoch in specific_epochs]
    row.append(f"{trainer.best_score:.5f}")
    data.append(row)
    
    # Create DataFrame with custom index label
    df = pd.DataFrame(data, columns=columns)
    df.index = ["Error: "]  # Set the index label
    
    # Save DataFrame to Excel
    df.to_excel(full_path_excel, sheet_name='Loss History', index=True)
    
    # Print DataFrame for immediate inspection
    print(df)
    
    return df

def create_detailed_report(trainer, model, path, experiment_name=None):
    """
    Creates a detailed training report with multiple metrics and saves it in various formats.
    
    Parameters:
    - trainer (Trainer): The trainer object with training history
    - model (NeuralODE): The trained model
    - path (str): Directory path for saving
    - experiment_name (str, optional): Name of the experiment for the report
    
    Returns:
    - report_df (DataFrame): The detailed report DataFrame
    """
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = experiment_name or f"experiment_{timestamp}"
    
    # Create directory for the report
    report_dir = os.path.join(path, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    # Collect training metrics
    loss_history = trainer.histories['loss_history']
    acc_history = trainer.histories.get('acc_history', [])
    val_loss_history = trainer.histories.get('val_loss_history', [])
    val_acc_history = trainer.histories.get('val_acc_history', [])
    time_per_epoch = trainer.histories.get('time_per_epoch', [])
    
    # Basic metrics
    metrics = {
        'Experiment Name': exp_name,
        'Total Epochs': len(loss_history),
        'Best Epoch': trainer.best_epoch,
        'Best Loss': trainer.best_score,
        'Perfect Classification': trainer.classif,
        'Early Stopping (No Improvement)': trainer.noimp,
        'Early Stopping (Relative Error)': trainer.relerr,
        'Early Stopping (Non-Convergence)': trainer.nonconv,
        'Model Architecture': model.architecture,
        'Activation Function': model.sigma,
        'Hidden Dimensions': model.hidden_dim,
        'Input Dimensions': model.input_dim,
        'Output Dimensions': model.output_dim,
        'Integration Time T': model.T,
        'Integration Method': model.method,
        'Number of Parameter Sets': model.num_vals,
        'Integration Step Size': model.dt,
        'Used Adjoint Method': model.adjoint,
        'Has Final Layer': model.final_layer,
        'Fixed Projector': trainer.fixed_projector,
        'Loss Function': trainer.loss_func,
    }
    
    # Compute additional metrics
    metrics.update({
        'Final Loss': loss_history[-1] if loss_history else None,
        'Final Accuracy': acc_history[-1] if acc_history else None,
        'Final Val Loss': val_loss_history[-1] if val_loss_history else None,
        'Final Val Accuracy': val_acc_history[-1] if val_acc_history else None,
        'Average Time per Epoch (s)': np.mean(time_per_epoch) if time_per_epoch else None,
        'Total Training Time (s)': np.sum(time_per_epoch) if time_per_epoch else None,
        'Loss Reduction (%)': ((loss_history[0] - trainer.best_score) / loss_history[0] * 100) if loss_history else None,
        'Epochs to Converge': trainer.best_epoch if trainer.best_epoch else len(loss_history),
    })
    
    # Create DataFrame for the report
    report_df = pd.DataFrame([metrics])
    
    # Export to various formats
    # CSV format
    csv_path = os.path.join(report_dir, f"{exp_name}_report.csv")
    report_df.to_csv(csv_path, index=False)
    
    # Excel format with multiple sheets
    excel_path = os.path.join(report_dir, f"{exp_name}_report.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        report_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add loss history sheet
        loss_df = pd.DataFrame({
            'Epoch': range(1, len(loss_history) + 1),
            'Train Loss': loss_history,
            'Val Loss': val_loss_history if val_loss_history else [None] * len(loss_history),
            'Train Accuracy': acc_history if acc_history else [None] * len(loss_history),
            'Val Accuracy': val_acc_history if val_acc_history else [None] * len(loss_history),
            'Time (s)': time_per_epoch if time_per_epoch else [None] * len(loss_history),
        })
        loss_df.to_excel(writer, sheet_name='Loss History', index=False)
    
    # Plot and save loss evolution
    if loss_history:
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', label='Train Loss')
        if val_loss_history:
            plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, 'r-', label='Val Loss')
        
        plt.axvline(x=trainer.best_epoch, color='g', linestyle='--', label=f'Best Epoch ({trainer.best_epoch})')
        
        plt.title('Loss Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(report_dir, f"{exp_name}_loss_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Detailed report saved to {report_dir}")
    return report_df

def model_comparison_table(models_results, path, filename='model_comparison'):
    """
    Creates a comparison table for multiple models with various metrics.
    
    Parameters:
    - models_results (list): List of dictionaries with model results
    - path (str): Directory path for saving
    - filename (str): Base filename for saving
    
    Returns:
    - comparison_df (DataFrame): The comparison table
    """
    # Create DataFrame from the results
    comparison_df = pd.DataFrame(models_results)
    
    # Save to Excel
    excel_path = os.path.join(path, f"{filename}.xlsx")
    comparison_df.to_excel(excel_path, index=False)
    
    # Save to CSV
    csv_path = os.path.join(path, f"{filename}.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    # Create a heatmap visualization for numeric columns
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        # Normalize the data for better visualization
        normalized_df = comparison_df[numeric_cols].copy()
        for col in numeric_cols:
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
        
        # Create heatmap
        sns.heatmap(normalized_df, annot=comparison_df[numeric_cols], fmt='.3g', cmap='viridis')
        plt.title('Model Comparison Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = os.path.join(path, f"{filename}_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print table for immediate inspection
    print(comparison_df)
    
    return comparison_df

def convergence_analysis(trainer, path, filename='convergence_analysis'):
    """
    Analyzes the convergence behavior of the training process.
    
    Parameters:
    - trainer (Trainer): The trainer object with training history
    - path (str): Directory path for saving
    - filename (str): Base filename for saving
    
    Returns:
    - analysis_df (DataFrame): DataFrame with convergence metrics
    """
    # Create directory for analysis
    analysis_dir = os.path.join(path, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Extract loss history
    loss_history = trainer.histories['loss_history']
    
    # Calculate convergence metrics
    epochs = len(loss_history)
    loss_changes = np.diff(loss_history)
    percent_changes = np.abs(loss_changes) / np.array(loss_history[:-1]) * 100
    
    # Create windows of convergence (e.g., first 10%, middle, last 10%)
    window_size = max(int(epochs * 0.1), 1)
    early_rate = np.mean(percent_changes[:window_size]) if window_size <= len(percent_changes) else np.nan
    middle_rate = np.mean(percent_changes[epochs//2-window_size//2:epochs//2+window_size//2]) if window_size <= len(percent_changes) else np.nan
    late_rate = np.mean(percent_changes[-window_size:]) if window_size <= len(percent_changes) else np.nan
    
    # Calculate convergence metrics
    convergence_metrics = {
        'Total Epochs': epochs,
        'Best Epoch': trainer.best_epoch,
        'Initial Loss': loss_history[0],
        'Final Loss': loss_history[-1],
        'Best Loss': trainer.best_score,
        'Loss Reduction (%)': ((loss_history[0] - trainer.best_score) / loss_history[0] * 100),
        'Average Percent Change Per Epoch (%)': np.mean(percent_changes) if len(percent_changes) > 0 else np.nan,
        'Early Convergence Rate (% change)': early_rate,
        'Middle Convergence Rate (% change)': middle_rate,
        'Late Convergence Rate (% change)': late_rate,
        'Ratio Early/Late Rate': early_rate / late_rate if late_rate != 0 else np.nan,
        'Perfect Classification': trainer.classif,
        'Early Stopping (No Improvement)': trainer.noimp,
        'Early Stopping (Relative Error)': trainer.relerr,
        'Early Stopping (Non-Convergence)': trainer.nonconv,
    }
    
    # Create DataFrame
    analysis_df = pd.DataFrame([convergence_metrics])
    
    # Save to Excel
    excel_path = os.path.join(analysis_dir, f"{filename}.xlsx")
    analysis_df.to_excel(excel_path, index=False)
    
    # Plot convergence rate over time
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    ax1 = plt.subplot(211)
    ax1.plot(range(1, epochs + 1), loss_history, 'b-')
    ax1.set_title('Loss Evolution')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot percent change
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(range(2, epochs + 1), percent_changes, 'r-')
    ax2.set_title('Percent Change in Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('% Change')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, f"{filename}_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis for immediate inspection
    print(analysis_df)
    
    return analysis_df

def export_hyperparameters(model, trainer, path, filename='hyperparameters'):
    """
    Exports the hyperparameters of the model and training process.
    
    Parameters:
    - model (NeuralODE): The model
    - trainer (Trainer): The trainer
    - path (str): Directory path for saving
    - filename (str): Base filename for saving
    
    Returns:
    - hyperparams_df (DataFrame): DataFrame with hyperparameters
    """
    # Collect hyperparameters
    hyperparams = {
        # Model hyperparameters
        'architecture': model.architecture,
        'activation_function': model.sigma,
        'hidden_dim': model.hidden_dim,
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'integration_time_T': model.T,
        'integration_method': model.method,
        'num_param_sets': model.num_vals,
        'step_size': model.dt,
        'adjoint_method': model.adjoint,
        'has_final_layer': model.final_layer,
        
        # Training hyperparameters
        'loss_function': trainer.loss_func,
        'fixed_projector': trainer.fixed_projector,
        'max_epochs': trainer.max_epochs,
        'patience': trainer.patience,
        'optimizer': trainer.optimizer.__class__.__name__,
    }
    
    # Create DataFrame
    hyperparams_df = pd.DataFrame([hyperparams])
    
    # Save to Excel and CSV
    hyperparams_df.to_excel(os.path.join(path, f"{filename}.xlsx"), index=False)
    hyperparams_df.to_csv(os.path.join(path, f"{filename}.csv"), index=False)
    
    # Also save as a readable text file
    with open(os.path.join(path, f"{filename}.txt"), 'w') as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    
    return hyperparams_df