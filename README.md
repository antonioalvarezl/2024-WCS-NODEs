# 2024-WCS-NODEs

Neural ODE-based Classification Framework with Controlled Dynamics

## Project Overview

This repository contains code from the article "Controlled cluster-based classification with neural ODEs," by Antonio Álvarez-López, Rafael Orive-Illera, and Enrique Zuazua.

### Mathematical Background

Consider the neural ODE:

$$
\dot{x} = w(t)\sigma(a(t) \cdot x + b(t)), \quad t \in (0, T).
$$

**Objective:**
For given $d\geq2$ and $N \geq 1$, find the minimum number of discontinuities $L\geq 0$ required by the control functions $(w,a,b)$ that allow the classification of any dataset $\{(x_i,y_i)\}_{i=1}^N$, where $x_i \sim U([0,1]^d)$ and $y_i \in \{0,1\}$ are randomly chosen for all $i$.

Classification is understood as finding some controls $(w,a,b)$ such that the flow map $\Phi_T$ of the neural ODE for some fixed $T>0$ satisfies:

$$
\Phi_T(x_i;w,a,b)^{(d)} > 1 \quad \text{for all } x_i \text{ such that } y_i = 1,
$$

and 

$$
\Phi_T(x_i;w,a,b)^{(d)} < 1 \quad \text{for all } x_i \text{ such that } y_i = 0.
$$

![Neural ODE Classification Animation](assets/trajectory.gif)

## Project Structure

The project is organized into several key components:

```
.
├── run.py                      # Main entry point for running experiments
├── Functions/                  # Core functionality modules
│   ├── create.py              # Dataset generation and utility functions
│   ├── neural_odes.py         # Neural ODE model implementation
│   ├── training.py            # Training loop and optimization
│   ├── gifs.py                # Trajectory visualization as GIFs
│   ├── plots.py               # Visualization of model behavior
│   ├── tables.py              # Results reporting and analysis
│   └── real_datasets.py       # Functions to load real-world datasets
├── Results/                   # Directory for storing experiment results
├── requirements.txt           # Dependencies for the project
└── README.md                  # This file
```

### Key Components

1. **Main Entry Point (`run.py`)**:
   - Provides a unified interface for running experiments
   - Handles command-line arguments and configuration
   - Coordinates the execution of individual or multiple experiments
   - Organizes results in a structured directory format

2. **Dataset Generation (`Functions/create.py`)**: 
   - Creates synthetic datasets (circles, blobs, moons, XOR patterns)
   - Utility functions for path creation and projector initialization
   - Functions for dataset handling and preprocessing

3. **Neural ODE Model (`Functions/neural_odes.py`)**:
   - Implementation of Neural ODEs with various architectures:
     - Inside: $\dot{x} = \sigma(W(t)x + b(t))$
     - Outside: $\dot{x} = W(t)\sigma(x) + b(t)$
     - Bottleneck: Encoding-decoding structure with nonlinearity
   - Support for different activation functions
   - Adjoint method for memory-efficient training

4. **Training Process (`Functions/training.py`)**:
   - Training loop with early stopping
   - Custom loss functions
   - Metrics tracking and model evaluation

5. **Visualization Tools**:
   - `Functions/gifs.py`: Creates animated visualizations of trajectories
   - `Functions/plots.py`: Generates static plots of decision boundaries and vector fields

6. **Results Analysis (`Functions/tables.py`)**:
   - Generates summary tables of results
   - Exports training metrics for analysis

7. **Real-world Datasets (`Functions/real_datasets.py`)**:
   - Functions to load and preprocess MNIST, CIFAR-10, and Fashion-MNIST
   - Dimensionality reduction for high-dimensional data

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA compatible GPU (optional but recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/2024-WCS-ClassifNODEs.git
   cd 2024-WCS-ClassifNODEs
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run a basic experiment with default parameters:

```bash
python run.py
```

### Custom Experiments

To run an experiment with custom parameters:

```bash
python run.py --dataset circles --noise 0.05 --hidden-dim 32 --architecture inside --activation tanh --epochs 5000
```

### Run Multiple Experiments

To run all predefined experiments:

```bash
python run.py --run-all
```

### Configuration Options

- `--dataset`: Dataset type (`circles`, `blobs`, `moons`, `xor`, `uniform`, `spiral`, `mnist`, `cifar10`, `fashion_mnist`)
- `--architecture`: Model architecture (`inside`, `outside`, `bottleneck`)
- `--activation`: Activation function (`tanh`, `relu`, `sigmoid`, `leakyrelu`, `2relu`, `trunrelu`)
- `--hidden-dim`: Hidden dimension size
- `--num-vals`: Number of parameter sets (L+1)
- `--integration-time`: Total integration time T
- `--method`: Integration method (`euler`, `rk4`, `dopri5`, `midpoint`)
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate for optimization
- `--epochs`: Maximum number of training epochs
- `--seed`: Random seed for reproducibility
- `--visualize`: Generate visualizations
- `--create-gif`: Create animated GIF of trajectories

### Example Workflow

1. Generate a synthetic dataset:
   ```python
   import torch
   from Functions.create import create_dataloader
   
   # Create a circles dataset with 1000 points and some noise
   train_loader, y, X0, X1 = create_dataloader(
       data_type='circles', 
       N=1000, 
       noise=0.05, 
       seed=42
   )
   ```

2. Initialize a Neural ODE model:
   ```python
   import torch.nn as nn
   from Functions.neural_odes import NeuralODE
   
   # Device configuration
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Create a fixed projector for the final layer
   projector = nn.Linear(2, 1)
   projector.weight.requires_grad = False
   projector.bias.requires_grad = False
   
   # Initialize model
   model = NeuralODE(
       device=device,
       fixed_projector=projector,
       input_dim=2,
       hidden_dim=32,
       output_dim=1,
       non_linearity='tanh',
       architecture='inside',
       T=10,
       num_vals=5,  # L+1 = 5 means L=4 discontinuities
       step_size=0.1,
       method='euler',
       final_layer=True
   )
   ```

3. Train the model:
   ```python
   import torch.optim as optim
   from Functions.training import Trainer
   
   # Create optimizer
   optimizer = optim.Adam(model.parameters(), lr=0.01)
   
   # Initialize trainer
   trainer = Trainer(
       model=model,
       optimizer=optimizer,
       device=device,
       fixed_projector=True,
       loss_func='bce',
       verbose=True
   )
   
   # Train the model
   trainer.train(
       datatrain=train_loader,
       max_epochs=3000,
       pathparams='./Results/my_experiment'
   )
   ```

4. Visualize results:
   ```python
   import torch
   from Functions.plots import levelsets, plot_data
   
   # Plot decision boundary
   levelsets(model, points=[X0, X1], path='./Results/my_experiment', fig_name='decision_boundary')
   
   # Visualize data points and trajectories
   plot_data(model, torch.cat([X0, X1]), torch.cat([torch.zeros(len(X0)), torch.ones(len(X1))]), 
            N=len(X0)+len(X1), path='./Results/my_experiment', final=True)
   ```

5. Create an animation of trajectories:
   ```python
   from Functions.gifs import traj_gif, select_random_samples
   
   # Select a subset of samples for visualization
   inputs, targets = select_random_samples(train_loader, num_samples=50)
   
   # Create trajectory animation
   gif_path = traj_gif(model, inputs, targets, path='./Results/my_experiment')
   print(f"Animation saved to {gif_path}")
   ```

6. Generate a report:
   ```python
   from Functions.tables import create_detailed_report
   
   # Create a detailed report of the training results
   report = create_detailed_report(trainer, model, path='./Results/my_experiment', experiment_name='circles_exp')
   ```

## Using the run.py Interface

The simplest way to run experiments is through the `run.py` interface:

```python
import run

# Define experiment configuration
config = {
    'dataset': 'circles',
    'architecture': 'inside',
    'activation': 'tanh',
    'hidden_dim': 32,
    'num_vals': 5,
    'epochs': 1000,
    'visualize': True,
    'create_gif': True
}

# Run the experiment
result = run.run_custom_experiment(config)

# Access model, trainer, and visualizations
model = result['model']
trainer = result['trainer']
viz_paths = result['viz_paths']

# Display model information
run.print_model_info(model)

# Display visualizations
run.display_visualizations(viz_paths)
```

## Real-world Datasets

To use real-world datasets:

```python
from Functions.real_datasets import load_mnist, load_cifar10, project_to_2d

# Load MNIST dataset with dimensionality reduction
train_loader, test_loader, X0, X1 = load_mnist(batch_size=64, pca_components=2)

# Visualize the 2D projection
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(X0[:, 0], X0[:, 1], c='blue', alpha=0.5, label='Class 0')
plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.5, label='Class 1')
plt.legend()
plt.title('MNIST 2D Projection')
plt.show()
```

## Results Directory Structure

All experiment results are saved in the `Results` directory with the following structure:

```
Results/
├── run_1/                     # First experiment
│   ├── dataset_architecture/  # Experiment-specific folder
│   │   ├── args.json          # Experiment arguments
│   │   ├── visualizations/    # Visualizations folder
│   │   │   ├── decision_boundary.png
│   │   │   ├── vector_field.png
│   │   │   ├── phase_portrait.png
│   │   │   ├── initial_points.png
│   │   │   ├── final_points.png
│   │   │   ├── loss_evolution.png
│   │   │   ├── trajectories.gif
│   │   │   └── classification_metrics.png
│   │   └── reports/           # Analysis reports
│   │       ├── detailed_report.xlsx
│   │       ├── convergence_analysis.csv
│   │       └── hyperparameters.txt
├── run_2/                     # Second experiment
│   └── ...
└── model_comparison/          # Model comparison folder
    ├── comparison_table.xlsx
    └── dataset_model_comparison.png
```

## Extended Features

### Vector Field Visualization

```python
from Functions.plots import plot_vector_field

# Visualize the vector field of the dynamics
fig, ax = plt.subplots(figsize=(8, 8))
plot_vector_field(model, ax=ax, t=5.0)
plt.savefig('./Results/vector_field.png')
```

### Model Comparison

```python
from run import compare_models

# Compare different models
results = [result1, result2, result3]  # Results from different experiments
comparison_dir = compare_models(results, "./Results")
```

### Convergence Analysis

```python
from Functions.tables import convergence_analysis

# Analyze convergence behavior
convergence_df = convergence_analysis(trainer, path='./Results/my_analysis')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The authors of the paper "Controlled cluster-based classification with neural ODEs"
- [PyTorch](https://pytorch.org/) team
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) library for ODE solvers
