"""
create.py - Dataset Generation and Utility Functions
---------------------------------------------------
This module provides functions for generating synthetic datasets and creating paths/projectors 
for Neural ODE based classification. It serves as the data preparation step of the workflow.

Key components:
- create_dataloader: Generates synthetic datasets (circles, blobs, moons, etc.)
- create_paths: Creates directory structures for saving results
- create_projector: Creates fixed or random linear projectors for model output

Workflow position: Initial step - data preparation
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

@torch.no_grad()
def create_dataloader(data_type, N=3000, d=2, noise=0, factor=0.15, shuffle=True, seed=1, rescale=False):
    """
    Generates a dataloader for various types of synthetic datasets.
    
    Parameters:
    - data_type (str): Type of dataset to generate ('circles', 'blobs', 'moons', 'xor', 'uniform', 'uniformbalanced').
    - N (int): Number of data points.
    - d (int): Dimensionality of the data.
    - noise (float): Amount of noise to add to the data.
    - factor (float): Scale factor (specific to 'circles' dataset).
    - shuffle (bool): Whether to shuffle the dataset.
    - seed (int): Random seed for reproducibility.
    - rescale (bool): If True, rescale data to have mean 0 and variance 1.
    
    Returns:
    - train (DataLoader): DataLoader object containing the training data.
    - y (Tensor): Labels of the dataset.
    - X0, X1 (Tensor): Subsets of the data corresponding to the two classes.
    """
    
    # Dictionary mapping dataset types to data generation functions
    # Each lambda creates a dataset with specific characteristics
    data_generators = {
        'circles': lambda: make_circles(N, noise=noise, factor=factor, random_state=seed, shuffle=shuffle),
        'blobs': lambda: make_blobs(n_samples=N, centers=np.array([[-1, -1], [1, 1]]), cluster_std=noise, random_state=seed),
        'moons': lambda: make_moons(N, noise=noise, shuffle=shuffle, random_state=seed),
        'xor': lambda: (X := torch.randint(low=0, high=2, size=(N, 2), dtype=torch.float32) + noise * torch.randn(N, 2),
                        (X[:, 0] > 0) ^ (X[:, 1] > 0)),
        'uniform': lambda: (X := torch.rand(N, d), torch.randint(0, 2, (N,)).float()),
        'uniformbalanced': lambda: ((X := torch.rand(N, d)), 
                                    (y := torch.cat([torch.zeros(N // 2), torch.ones(N // 2)])[torch.randperm(N)]))
    }
    
    # Generate the dataset
    if data_type in data_generators:
        X, y = data_generators[data_type]()
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Optionally rescale the dataset to standardize features
    if rescale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Split the dataset into two classes for visualization and analysis
    mask_0, mask_1 = (y == 0), (y == 1)
    X0, X1 = X[mask_0], X[mask_1]
    
    # Convert data to tensors for use with PyTorch
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = y.clone().detach() if data_type == 'uniform' else torch.tensor(y, dtype=torch.long)
    
    # Create a TensorDataset and DataLoader for batched training
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(train_data, batch_size=N, shuffle=shuffle, generator=g)
    
    return train, y, X0, X1

def create_paths(non_linearity, architecture, d, hidden_dim, num_vals, distribution, N, seed_data, seed_params, rescale):
    """
    Creates and returns a unique directory path for saving results.
    This function establishes a consistent naming convention for experiments.
    
    Parameters:
    - non_linearity (str): Non-linearity used in the model.
    - architecture (str): Model architecture.
    - d (int): Data dimensionality.
    - hidden_dim (int): Number of hidden dimensions.
    - num_vals (int): Number of values in the control.
    - distribution (str): Type of data distribution.
    - N (int): Number of data points.
    - seed_data, seed_params (int): Seeds for data generation and model parameters.
    - rescale (bool): Indicates if data was rescaled.
    
    Returns:
    - full_path (str): Unique path where results will be stored.
    """
    
    # Create base directory structure
    subfolder = os.path.join('Results', distribution) if distribution != 'uniform' else 'Results'
    os.makedirs(subfolder, exist_ok=True)
    
    # Build directory name based on experiment parameters
    suffix = "_Rescaled" if rescale else ""
    folder_path = os.path.join(subfolder,
                               f"Sigma={non_linearity}_Arch={architecture}_Width={hidden_dim}",
                               f"2N={N}_Dim={d}_L={num_vals-1}{suffix}")
    os.makedirs(folder_path, exist_ok=True)

    # Ensure a unique directory name by incrementing counter
    i = 1
    while True:
        full_path = os.path.join(folder_path, f"DS{seed_data}PS{seed_params}Results{i}")
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            break
        i += 1

    return full_path

def create_projector(device, input_dim, output_dim, random_projector=True):
    """
    Creates a fixed or random linear projector for use in a neural network.
    This projector serves as the final layer for classification.
    
    Parameters:
    - device (torch.device): The device (CPU or GPU) on which to allocate the projector.
    - input_dim (int): Dimensionality of the input data.
    - output_dim (int): Dimensionality of the output data.
    - random_projector (bool): If True, initialize the projector with random weights and bias.
    
    Returns:
    - projector (nn.Linear): A linear layer with fixed weights and bias.
    """
    
    projector = nn.Linear(input_dim, output_dim)
    projector.to(device)
    
    with torch.no_grad():
        if random_projector:
            # Initialize with random weights and bias
            projector.weight.copy_(torch.randn(output_dim, input_dim))
            projector.bias.copy_(torch.randn(output_dim))
        elif output_dim > 1:
            # Identity projection for multi-dimensional output
            projector.weight.copy_(torch.eye(output_dim, input_dim))
            projector.bias.fill_(1.0)
        else:
            # Binary classification projection
            weight = torch.zeros((output_dim, input_dim))
            weight[-1, -1] = 1
            projector.weight.copy_(weight)
            projector.bias.copy_(torch.tensor([-0.5]))

    # Disable gradient computation for the projector parameters
    projector.weight.requires_grad = False
    projector.bias.requires_grad = False
    
    return projector

# Extended functionality - Additional dataset types
def create_spiral_dataset(N=1000, noise=0.1, turns=2, seed=1):
    """
    Creates a spiral dataset for binary classification.
    
    Parameters:
    - N (int): Total number of points
    - noise (float): Amount of noise to add
    - turns (float): Number of spiral turns
    - seed (int): Random seed for reproducibility
    
    Returns:
    - X (Tensor): Feature matrix
    - y (Tensor): Labels
    """
    np.random.seed(seed)
    N = N // 2  # Points per class
    
    # Generate spiral data
    theta = np.sqrt(np.random.rand(N)) * turns * np.pi
    
    # Spiral class 1
    r_a = 2 * theta + np.pi
    x_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    
    # Spiral class 2
    r_b = 2 * theta
    x_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    
    X = np.vstack([x_a, x_b])
    y = np.hstack([np.zeros(N), np.ones(N)])
    
    # Add noise
    X += np.random.randn(N*2, 2) * noise
    
    # Shuffle the data
    indices = np.random.permutation(2*N)
    X, y = X[indices], y[indices]
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return X_tensor, y_tensor

def train_test_split(dataloader, test_size=0.2, seed=42):
    """
    Splits a dataloader into training and test sets.
    
    Parameters:
    - dataloader (DataLoader): The dataloader to split
    - test_size (float): Proportion of data to use for testing
    - seed (int): Random seed for reproducibility
    
    Returns:
    - train_loader, test_loader (DataLoader): Training and test dataloaders
    """
    # Get the dataset from the dataloader
    dataset = dataloader.dataset
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size
    
    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create new dataloaders
    batch_size = dataloader.batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader