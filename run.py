"""
run.py - Main entry point for Neural ODE Classification
------------------------------------------------------
This script organizes and runs experiments with Neural ODEs for classification tasks.
It provides functions to run individual or multiple experiments with different
configurations and visualize results.

Usage examples:
- From command line: python run.py --run-all
- From a notebook: import run; run.run_experiment(...)
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from IPython.display import display, Image, HTML
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from Functions.create import create_dataloader, create_spiral_dataset, train_test_split
from Functions.neural_odes import NeuralODE, RegularizedDynamics
from Functions.training import Trainer, create_optimizer, create_lr_scheduler, compute_metrics
from Functions.plots import (levelsets, plot_data, plot_vector_field, plot_phase_portrait, 
                  loss_evolution, plot_logloss, visualize_model_comparison)
from Functions.gifs import traj_gif, select_random_samples, create_trajectory_comparison
from Functions.tables import dataframe, create_detailed_report, model_comparison_table, convergence_analysis
from Functions.real_datasets import load_mnist, load_cifar10, load_fashion_mnist, project_to_2d


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Neural ODE Classification')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='circles',
                        choices=['circles', 'blobs', 'moons', 'xor', 'uniform', 
                                'uniformbalanced', 'spiral', 'mnist', 'cifar10', 'fashion_mnist'],
                        help='Dataset type')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples for synthetic datasets')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise level for synthetic datasets') 
    parser.add_argument('--rescale', action='store_true', default=False,
                        help='Rescale data to have mean 0 and variance 1')
    parser.add_argument('--pca-components', type=int, default=None,
                        help='Reduce dimensions using PCA (for real datasets)')

    # Model parameters
    parser.add_argument('--architecture', type=str, default='inside',
                        choices=['inside', 'outside', 'bottleneck'],
                        help='Model architecture')
    parser.add_argument('--activation', type=str, default='tanh', 
                        choices=['tanh', 'relu', 'sigmoid', 'leakyrelu', '2relu', 'trunrelu'],
                        help='Activation function')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden dimension size')
    parser.add_argument('--input-dim', type=int, default=2,
                        help='Input dimension size')
    parser.add_argument('--output-dim', type=int, default=1, 
                        help='Output dimension size')
    parser.add_argument('--num-vals', type=int, default=5,
                        help='Number of parameter sets (L+1)')
    parser.add_argument('--integration-time', type=float, default=10.0,
                        help='Total integration time T')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Integration step size')
    parser.add_argument('--method', type=str, default='euler',
                        choices=['euler', 'rk4', 'dopri5', 'midpoint'],
                        help='Integration method')
    parser.add_argument('--adjoint', action='store_true', default=False,
                        help='Use adjoint method for optimization')
    parser.add_argument('--regularize', action='store_true', default=False,
                        help='Apply L2 regularization to dynamics')

    # Training parameters 
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw', 'rmsprop'],
                        help='Optimizer type')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--loss-function', type=str, default='bce',
                        choices=['mse', 'bce', 'cross_entropy', 'linear_sep'],
                        help='Loss function')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=1000,
                        help='Patience for early stopping')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                        choices=[None, 'plateau', 'step', 'cosine'],
                        help='Learning rate scheduler')

    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Generate visualizations')
    parser.add_argument('--create-gif', action='store_true', default=False,
                        help='Create animated GIF of trajectories')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print detailed logs')
    parser.add_argument('--run-all', action='store_true', default=False,
                        help='Run all predefined experiments')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Display plots (used in notebook)')
    
    if 'ipykernel_launcher' in sys.argv[0]:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args

def setup_experiment(args, run_id=None):
    """Setup experiment directories and settings."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory with run_id format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base directory is Results
    base_dir = "Results"
    
    # If run_id is provided, use it for the subfolder
    if run_id is not None:
        output_dir = os.path.join(base_dir, f"run_{run_id}")
    else:
        # Find the next available run_id
        i = 1
        while True:
            output_dir = os.path.join(base_dir, f"run_{i}")
            if not os.path.exists(output_dir):
                break
            i += 1
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add experiment specific subfolder
    exp_name = args.exp_name or f"{args.dataset}_{args.architecture}_{timestamp}"
    experiment_dir = os.path.join(output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Experiment: {exp_name}")
    print(f"Output directory: {experiment_dir}")
    print(f"Device: {device}")
    
    return device, experiment_dir


def load_dataset(args, device):
    """Load dataset based on arguments."""
    if args.dataset in ['circles', 'blobs', 'moons', 'xor', 'uniform', 'uniformbalanced']:
        # Synthetic dataset
        print(f"Creating synthetic dataset: {args.dataset}")
        train_loader, y, X0, X1 = create_dataloader(
            data_type=args.dataset,
            N=args.num_samples,
            noise=args.noise,
            rescale=args.rescale,
            seed=args.seed
        )
        # Split into train and test sets
        test_loader, _ = train_test_split(train_loader, test_size=0.2, seed=args.seed)
    elif args.dataset == 'spiral':
        # Spiral dataset (using the additional function)
        print("Creating spiral dataset")
        X, y = create_spiral_dataset(N=args.num_samples, noise=args.noise, seed=args.seed)
        
        # Split into 0 and 1 classes for visualization
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        # Create dataloader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Split into train and test sets
        test_loader, _ = train_test_split(train_loader, test_size=0.2, seed=args.seed)
    elif args.dataset == 'mnist':
        # MNIST dataset
        print("Loading MNIST dataset")
        train_loader, test_loader, X0, X1 = load_mnist(
            batch_size=args.batch_size,
            pca_components=args.pca_components or args.input_dim,
            seed=args.seed
        )
    elif args.dataset == 'cifar10':
        # CIFAR-10 dataset
        print("Loading CIFAR-10 dataset")
        train_loader, test_loader, X0, X1 = load_cifar10(
            batch_size=args.batch_size,
            pca_components=args.pca_components or args.input_dim,
            seed=args.seed
        )
    elif args.dataset == 'fashion_mnist':
        # Fashion-MNIST dataset
        print("Loading Fashion-MNIST dataset")
        train_loader, test_loader, X0, X1 = load_fashion_mnist(
            batch_size=args.batch_size,
            pca_components=args.pca_components or args.input_dim,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Dataset loaded: {len(train_loader.dataset)} training samples")
    if test_loader:
        print(f"                {len(test_loader.dataset)} test samples")
    
    return train_loader, test_loader, X0, X1

def create_model(args, device):
    """Create Neural ODE model."""
    # Create fixed projector for final layer
    fixed_projector = True
    projector = nn.Linear(args.input_dim, args.output_dim)
    projector.weight.requires_grad = False
    projector.bias.requires_grad = False
    
    with torch.no_grad():
        # Initialize with specific values for reproducibility
        if args.output_dim == 1:
            weight = torch.zeros((args.output_dim, args.input_dim))
            weight[-1, -1] = 1.0
            projector.weight.copy_(weight)
            projector.bias.copy_(torch.tensor([-0.5]))
        else:
            projector.weight.copy_(torch.eye(args.output_dim, args.input_dim))
            projector.bias.fill_(0.0)
    
    # Check if regularization is requested
    if args.regularize:
        print("Using regularized dynamics")
        # Using a custom dynamics class with regularization
        dynamics_class = RegularizedDynamics
    else:
        dynamics_class = None  # Use default dynamics
    
    # Create model
    model = NeuralODE(
        device=device,
        fixed_projector=projector,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        non_linearity=args.activation,
        architecture=args.architecture,
        T=args.integration_time,
        num_vals=args.num_vals,
        step_size=args.step_size,
        method=args.method,
        adjoint=args.adjoint,
        final_layer=True,
        seed_params=args.seed,
        dynamics_class=dynamics_class,
        reg_lambda=args.weight_decay if args.regularize else 0.0
    )
    
    print(f"Model created:")
    print(f"  Architecture: {args.architecture}")
    print(f"  Activation: {args.activation}")
    print(f"  Hidden dimensions: {args.hidden_dim}")
    print(f"  Number of parameter sets (L+1): {args.num_vals}")
    print(f"  Integration time T: {args.integration_time}")
    print(f"  Integration method: {args.method}")
    print(f"  Using adjoint method: {args.adjoint}")
    print(f"  Using regularization: {args.regularize}")
    
    return model, fixed_projector


def train_model(args, model, train_loader, test_loader, fixed_projector, device, output_dir):
    """Train the model."""
    # Create optimizer
    optimizer = create_optimizer(
        model, 
        optimizer_name=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    lr_scheduler = None
    if args.lr_scheduler:
        lr_scheduler = create_lr_scheduler(
            optimizer,
            scheduler_name=args.lr_scheduler,
            patience=args.patience // 10,
            factor=0.5,
            min_lr=args.learning_rate * 0.01
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        fixed_projector=fixed_projector,
        loss_func=args.loss_function,
        verbose=args.verbose
    )
    
    print(f"Starting training:")
    print(f"  Optimizer: {args.optimizer} (lr={args.learning_rate})")
    print(f"  Loss function: {args.loss_function}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  LR scheduler: {args.lr_scheduler or 'None'}")
    
    # Train the model
    start_time = time.time()
    trainer.train(
        datatrain=train_loader,
        max_epochs=args.epochs,
        pathparams=output_dir,
        validation_data=test_loader,
        patience=args.patience,
        lr_scheduler=lr_scheduler
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"  Best epoch: {trainer.best_epoch}")
    print(f"  Best loss: {trainer.best_score:.6f}")
    
    return trainer

# Modify the beginning of the visualize_results function in run.py:
def visualize_results(args, model, trainer, train_loader, test_loader, X0, X1, output_dir):
    """Generate comprehensive visualizations of the results."""
    print("Generating visualizations...")
    
    # Create visualization subdirectory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Force visualization and GIF creation regardless of args
    args.visualize = True
    args.create_gif = True
    
    # Dictionary to store all visualization paths
    viz_paths = {}
    
    # Plot decision boundary
    if args.input_dim == 2:  # Only for 2D data
        print("  Creating decision boundary plot...")
        # Use levelsets function to create the decision boundary plot
        fig, ax = levelsets(model, points=[X0, X1], path=viz_dir, fig_name='decision_boundary', return_fig=True)
        viz_paths['decision_boundary'] = os.path.join(viz_dir, 'decision_boundary.png')
        
        # Show the plot if requested
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # Visualize vector field
        print("  Creating vector field visualization...")
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_vector_field(model, ax=ax, t=args.integration_time * 0.5)
        plt.savefig(os.path.join(viz_dir, 'vector_field.png'), dpi=300, bbox_inches='tight')
        viz_paths['vector_field'] = os.path.join(viz_dir, 'vector_field.png')
        
        # Show the plot if requested
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # Phase portrait with trajectories
        print("  Creating phase portrait...")
        fig, ax = plt.subplots(figsize=(8, 8))
        # Select some samples for trajectories
        inputs, targets = select_random_samples(train_loader, num_samples=20)
        with torch.no_grad():
            traj = model.flow.trajectory(inputs, int(model.T / model.dt) + 1)
        plot_phase_portrait(model, traj, targets, ax=ax)
        plt.savefig(os.path.join(viz_dir, 'phase_portrait.png'), dpi=300, bbox_inches='tight')
        viz_paths['phase_portrait'] = os.path.join(viz_dir, 'phase_portrait.png')
        
        # Show the plot if requested
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    # Plot data points and final state
    
    print("  Creating data visualization...")
    if isinstance(X0, np.ndarray):
        X0 = torch.from_numpy(X0).float()
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1).float()

    # Limitar el número de muestras por clase (usar min para evitar errores si hay menos de 500 muestras)
    max_samples = min(500, len(X0), len(X1))
    all_inputs = torch.cat([X0[:max_samples], X1[:max_samples]])
    all_targets = torch.cat([torch.zeros(max_samples), torch.ones(max_samples)])
    
    # Initial state
    initial_plot = plot_data(model, all_inputs, all_targets, N=len(all_inputs), 
              path=viz_dir, init=True, rescale=args.rescale)
    viz_paths['initial_state'] = initial_plot
    
    # Final state
    final_plot = plot_data(model, all_inputs, all_targets, N=len(all_inputs), 
              path=viz_dir, final=True, rescale=args.rescale)
    viz_paths['final_state'] = final_plot
    
    # Show the plots if requested
    if args.show_plots:
        # Display initial state
        display(Image(filename=initial_plot))
        # Display final state
        display(Image(filename=final_plot))
    
    # Plot loss evolution
    print("  Creating loss evolution plots...")
    if hasattr(trainer, 'histories') and 'loss_history' in trainer.histories:
        loss_path = os.path.join(viz_dir, 'loss_evolution.png')
        loss_evolution(trainer, len(trainer.histories['loss_history'])-1, 
                      viz_dir, filename='loss_evolution.png')
        viz_paths['loss_evolution'] = loss_path
        
        # Show the plot if requested
        if args.show_plots:
            display(Image(filename=loss_path))
        
        # Log loss plot
        log_loss_path = os.path.join(viz_dir, 'LossVsEpochs.png')
        plot_logloss(trainer, viz_dir, export_fig=True)
        viz_paths['log_loss'] = log_loss_path
        
        # Show the plot if requested
        if args.show_plots:
            display(Image(filename=log_loss_path))
    
    # Create trajectory animation
    if args.create_gif:  # Removed the input_dim condition to ensure it tries to create the GIF
        print("  Creating trajectory animation...")
        try:
            # Select a subset of samples for visualization
            inputs, targets = select_random_samples(train_loader, num_samples=50)
            # Create trajectory animation
            gif_path = traj_gif(model, inputs, targets, path=viz_dir)
            viz_paths['trajectory_gif'] = gif_path
            print(f"  Animation saved to {os.path.basename(gif_path)}")
            
            # Show the GIF if requested
            if args.show_plots:
                display(HTML(f'<img src="{gif_path}" alt="Trajectory Animation" loop autoplay>'))
        except Exception as e:
            print(f"  Warning: Could not create GIF animation: {str(e)}")
    
    # If we have a test set, create classification visualizations
    if test_loader is not None:
        print("  Creating classification metrics visualization...")
        metrics = compute_metrics(model, test_loader)
        
        # Create a bar chart of metrics
        plt.figure(figsize=(10, 6))
        metrics_values = [metrics[name] for name in metrics]
        metrics_names = list(metrics.keys())
        plt.bar(metrics_names, metrics_values, color='skyblue')
        plt.ylim(0, 1.0)
        plt.title('Classification Metrics')
        metrics_path = os.path.join(viz_dir, 'classification_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        viz_paths['metrics'] = metrics_path
        
        # Show the plot if requested
        if args.show_plots:
            plt.show()
        else:
            plt.close()
        
        # Save metrics to a CSV file
        import pandas as pd
        pd.DataFrame([metrics]).to_csv(os.path.join(viz_dir, 'metrics.csv'), index=False)
    
    print("Visualizations completed")
    
    return viz_paths, viz_dir

def generate_report(args, trainer, model, output_dir):
    """Generate detailed report of the experiment."""
    print("Generating detailed report...")
    
    # Create reports subdirectory
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create a detailed report
    exp_name = args.exp_name or f"{args.dataset}_{args.architecture}"
    report = create_detailed_report(trainer, model, reports_dir, experiment_name=exp_name)
    
    # Create convergence analysis
    conv_analysis = convergence_analysis(trainer, reports_dir, filename=f"{exp_name}_convergence")
    
    # Create loss summary table
    loss_df = dataframe(trainer, reports_dir)
    
    # Create hyperparameter export
    from Functions.tables import export_hyperparameters
    hyperparams_df = export_hyperparameters(model, trainer, reports_dir, filename=f"{exp_name}_hyperparams")
    
    print("Report generation completed")
    
    return reports_dir

def compare_models(results, output_dir):
    """Compare multiple models and create comparative visualizations."""
    print("\n" + "="*80)
    print("COMPARING MODELS")
    print("="*80)
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, 'model_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract models and results
    models = {f"{r['dataset']}_{r['architecture']}": r['model'] for r in results}
    
    # Create a comparison table
    comparison_data = [{
        'Dataset': r['dataset'],
        'Architecture': r['architecture'],
        'Activation': r['activation'],
        'Num Parameter Sets': r['num_vals'],
        'Best Loss': r['best_score'],
        'Output Directory': os.path.basename(r['output_dir'])
    } for r in results]
    
    # Generate comparison table
    model_comparison_table(comparison_data, comparison_dir)
    
    # Generate comparison visualizations for compatible models (same dataset)
    for dataset in set(r['dataset'] for r in results):
        # Get models for this dataset
        dataset_models = {f"{r['architecture']}_{r['activation']}": r['model'] 
                         for r in results if r['dataset'] == dataset}
        
        if len(dataset_models) > 1:
            print(f"  Creating comparison visualization for {dataset} dataset...")
            # Get a sample loader for visualization
            for r in results:
                if r['dataset'] == dataset:
                    # Load a small sample of the dataset
                    if dataset in ['circles', 'blobs', 'moons', 'xor', 'uniform', 'spiral']:
                        train_loader, _, X0, X1 = load_dataset(
                            argparse.Namespace(**{'dataset': dataset, 'num_samples': 500, 
                                              'noise': 0.05, 'rescale': False, 'seed': 42,
                                              'batch_size': 100, 'pca_components': None})
                        , 'cpu')
                    else:
                        # For real datasets, use a smaller sample
                        train_loader, _, _, _ = load_dataset(
                            argparse.Namespace(**{'dataset': dataset, 'batch_size': 100, 
                                              'pca_components': 2, 'seed': 42})
                        , 'cpu')
                    break
            
            # Create comparison figure
            fig = visualize_model_comparison(dataset_models, train_loader, 
                                           path=comparison_dir, 
                                           filename=f"{dataset}_model_comparison.png")
            
            # Display comparison figure
            plt.figure(figsize=(12, 8))
            plt.imshow(plt.imread(os.path.join(comparison_dir, f"{dataset}_model_comparison.png")))
            plt.axis('off')
            plt.title(f'Comparison of models on {dataset} dataset')
            plt.show()
    
    print(f"Model comparison completed. Results saved to {comparison_dir}")
    
    return comparison_dir

def run_all_experiments(args=None, show_plots=True):
    """Run all predefined experiments."""
    print("Running all predefined experiments...")
    
    # Get predefined experiments
    experiments = get_predefined_experiments()
    
    # Create base output directory
    base_dir = "Results"
    os.makedirs(base_dir, exist_ok=True)
    
    # If args is provided, use it as base for all experiments
    base_args = args or parse_args()
    
    # Set show_plots flag for all experiments
    for exp in experiments:
        exp['show_plots'] = show_plots
    
    # Run each experiment
    results = []
    for i, exp_args in enumerate(experiments):
        # Create namespace with default values and override with experiment values
        exp_namespace = argparse.Namespace(**vars(base_args))
        # Update with experiment-specific values
        for key, value in exp_args.items():
            setattr(exp_namespace, key, value)
        
        # Run the experiment
        result = run_experiment(exp_namespace, run_id=i+1, return_results=True)
        results.append(result)
    
    # Compare models
    comparison_dir = compare_models(results, base_dir)
    
    print("\nAll experiments completed successfully.")
    print(f"Results are saved in the {base_dir} directory.")
    
    return results, comparison_dir

def print_model_info(model):
    """Print detailed information about the model."""
    print(f"Model Architecture: {model.architecture}")
    print(f"Input Dimension: {model.input_dim}")
    print(f"Hidden Dimension: {model.hidden_dim}")
    print(f"Output Dimension: {model.output_dim}")
    print(f"Activation Function: {model.sigma}")
    print(f"Number of Parameter Sets (L+1): {model.num_vals}")
    print(f"Integration Time (T): {model.T}")
    print(f"Integration Method: {model.method}")
    print(f"Step Size: {model.dt}")
    print(f"Using Adjoint Method: {model.adjoint}")
    print(f"Has Final Layer: {model.final_layer}")

def display_visualizations(viz_paths):
    """Display all visualizations from a result."""
    if not viz_paths:
        print("No visualizations available.")
        return
    
    # Display static images first
    for name, path in viz_paths.items():
        if name != 'trajectory_gif':
            print(f"\n{name.replace('_', ' ').title()}:")
            display(Image(filename=path))
    
    # Display GIF if available
    if 'trajectory_gif' in viz_paths:
        print("\nTrajectory Animation:")
        display(HTML(f'<img src="{viz_paths["trajectory_gif"]}" alt="Trajectory Animation" loop autoplay>'))

def run_custom_experiment(config, show_plots=True):
    """Run a custom experiment with the given configuration."""
    # Get default args first
    default_args = parse_args()
    
    # Create a new namespace with all defaults
    args = argparse.Namespace(**vars(default_args))
    
    # Now update with the user's config
    for key, value in config.items():
        setattr(args, key, value)
    
    # Set show_plots flag
    args.show_plots = show_plots
    
    # Run the experiment
    result = run_experiment(args, return_results=True)
    
    return result



def run_experiment(args, run_id=None, return_results=False):
    """Run a single experiment with the given arguments."""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {args.dataset} dataset with {args.architecture} architecture")
    print("="*80)
    
    # Setup experiment
    device, output_dir = setup_experiment(args, run_id)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    train_loader, test_loader, X0, X1 = load_dataset(args, device)
    
    # Create model
    model, fixed_projector = create_model(args, device)
    
    # Train model
    trainer = train_model(args, model, train_loader, test_loader, fixed_projector, device, output_dir)
    
    # Visualize results
    viz_paths, viz_dir = visualize_results(args, model, trainer, train_loader, test_loader, X0, X1, output_dir)
    
    # Generate report
    reports_dir = generate_report(args, trainer, model, output_dir)
    
    print(f"Experiment completed. Results saved to {output_dir}")
    if viz_paths:
        print("\nVisualizations saved:")
        for name, path in viz_paths.items():
            print(f"  - {name}: {path}")
        
    # Save model
    model_path = os.path.join(output_dir, f"{args.dataset}_{args.architecture}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_epoch': trainer.best_epoch,
        'best_score': trainer.best_score,
        'args': vars(args)
    }, model_path)
    
    if return_results:
        return {
            'model': model,
            'trainer': trainer,
            'output_dir': output_dir,
            'best_score': trainer.best_score,
            'dataset': args.dataset,
            'architecture': args.architecture,
            'num_vals': args.num_vals,
            'activation': args.activation,
            'viz_paths': viz_paths
        }
    else:
        return output_dir

def get_predefined_experiments():
    """Return a list of predefined experiment configurations."""
    # Define a set of common experiments
    experiments = [
        # Circles dataset with different architectures
        {
            'dataset': 'circles', 
            'architecture': 'inside', 
            'activation': 'tanh',
            'num_vals': 5,
            'epochs': 1000,
            'visualize': True,
            'create_gif': True,
            'show_plots': True
        },
        {
            'dataset': 'circles', 
            'architecture': 'outside', 
            'activation': 'tanh',
            'num_vals': 5,
            'epochs': 1000,
            'visualize': True,
            'show_plots': True
        },
        {
            'dataset': 'circles', 
            'architecture': 'bottleneck', 
            'activation': 'tanh',
            'num_vals': 5,
            'hidden_dim': 64,
            'epochs': 1000,
            'visualize': True,
            'show_plots': True
        },
        
        # Moons dataset
        {
            'dataset': 'moons', 
            'architecture': 'inside', 
            'activation': 'tanh',
            'num_vals': 5,
            'epochs': 1500,
            'visualize': True,
            'create_gif': True,
            'show_plots': True
        },
        
        # XOR dataset (harder problem)
        {
            'dataset': 'xor', 
            'architecture': 'bottleneck', 
            'activation': 'tanh',
            'num_vals': 10,  # More parameter sets for harder problem
            'hidden_dim': 64,
            'epochs': 2000,
            'visualize': True,
            'show_plots': True
        },
        
        # MNIST with dimensionality reduction
        {
            'dataset': 'mnist',
            'architecture': 'bottleneck',
            'activation': 'relu',
            'input_dim': 2,
            'pca_components': 2,
            'hidden_dim': 64,
            'num_vals': 10,
            'epochs': 1000,
            'batch_size': 256,
            'visualize': True,
            'show_plots': True
        },
        
        # Spiral dataset
        {
            'dataset': 'spiral',
            'architecture': 'inside',
            'activation': 'tanh',
            'num_vals': 8,
            'epochs': 1500,
            'visualize': True,
            'create_gif': True,
            'show_plots': True
        }
    ]
    
    return experiments

def main():
    """Main entry point when run as a script."""
    args = parse_args()
    
    # If run-all flag is set, run all predefined experiments
    if args.run_all:
        results, comparison_dir = run_all_experiments(args)
        print(f"Compared {len(results)} models. Results saved to {comparison_dir}")
    else:
        # Run a single experiment with the provided arguments
        output_dir = run_experiment(args)
        print(f"Experiment completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main()