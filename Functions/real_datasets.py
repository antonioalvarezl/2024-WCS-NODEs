"""
Functions for loading real-world datasets (MNIST, CIFAR) for Neural ODE classification.
These functions complement the synthetic dataset generators in the main module.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA

def load_mnist(batch_size=100, train_ratio=0.8, flatten=True, pca_components=None, seed=42):
    """
    Load the MNIST dataset and prepare it for Neural ODE classification.
    
    Parameters:
    - batch_size (int): Size of batches for dataloader
    - train_ratio (float): Ratio of data to use for training (remaining for testing)
    - flatten (bool): Whether to flatten the images or keep 2D structure
    - pca_components (int, optional): If provided, reduce dimensions using PCA
    - seed (int): Random seed for reproducibility
    
    Returns:
    - train_loader (DataLoader): Training data loader
    - test_loader (DataLoader): Test data loader
    - X0 (Tensor): Features for class 0
    - X1 (Tensor): Features for class 1
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create full dataset by combining train and test
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    # Extract tensors and labels
    all_tensors = []
    all_labels = []
    
    for img, label in full_dataset:
        if flatten:
            # Flatten the image
            img = img.view(-1)
        all_tensors.append(img)
        all_labels.append(label)
    
    # Convert to tensors
    X = torch.stack(all_tensors)
    y = torch.tensor(all_labels)
    
    # Apply PCA if requested
    if pca_components is not None and pca_components < X.shape[1]:
        X_np = X.numpy()
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X_np)
        X = torch.tensor(X_reduced, dtype=torch.float32)
        print(f"Reduced dimensions from {X_np.shape[1]} to {pca_components} using PCA")
        print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Convert to binary classification (even vs odd numbers)
    binary_labels = (y % 2 == 0).long()
    
    # Create dataset
    dataset = TensorDataset(X, binary_labels)
    
    # Split into train and test
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract samples for each class for visualization
    X0 = []
    X1 = []
    
    for features, label in dataset:
        if label == 0 and len(X0) < 500:  # Limit to 500 samples per class
            X0.append(features)
        elif label == 1 and len(X1) < 500:
            X1.append(features)
        
        # Break if we have enough samples
        if len(X0) >= 500 and len(X1) >= 500:
            break
    
    X0 = torch.stack(X0)
    X1 = torch.stack(X1)
    
    return train_loader, test_loader, X0, X1

def load_cifar10(batch_size=100, train_ratio=0.8, flatten=True, pca_components=None, binary_classes=(0, 1), seed=42):
    """
    Load the CIFAR-10 dataset and prepare it for Neural ODE classification.
    
    Parameters:
    - batch_size (int): Size of batches for dataloader
    - train_ratio (float): Ratio of data to use for training (remaining for testing)
    - flatten (bool): Whether to flatten the images or keep 2D structure
    - pca_components (int, optional): If provided, reduce dimensions using PCA
    - binary_classes (tuple): Two class indices to use for binary classification
    - seed (int): Random seed for reproducibility
    
    Returns:
    - train_loader (DataLoader): Training data loader
    - test_loader (DataLoader): Test data loader
    - X0 (Tensor): Features for class 0
    - X1 (Tensor): Features for class 1
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create full dataset by combining train and test
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    # Extract tensors and labels
    all_tensors = []
    all_labels = []
    
    for img, label in full_dataset:
        if flatten:
            # Flatten the image
            img = img.view(-1)
        
        # Filter for binary classification (keep only the specified classes)
        if label in binary_classes:
            all_tensors.append(img)
            # Convert to binary label (0 or 1)
            binary_label = 0 if label == binary_classes[0] else 1
            all_labels.append(binary_label)
    
    # Convert to tensors
    X = torch.stack(all_tensors)
    y = torch.tensor(all_labels)
    
    # Apply PCA if requested
    if pca_components is not None and pca_components < X.shape[1]:
        X_np = X.numpy()
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X_np)
        X = torch.tensor(X_reduced, dtype=torch.float32)
        print(f"Reduced dimensions from {X_np.shape[1]} to {pca_components} using PCA")
        print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Split into train and test
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract samples for each class for visualization
    X0 = []
    X1 = []
    
    for features, label in dataset:
        if label == 0 and len(X0) < 500:  # Limit to 500 samples per class
            X0.append(features)
        elif label == 1 and len(X1) < 500:
            X1.append(features)
        
        # Break if we have enough samples
        if len(X0) >= 500 and len(X1) >= 500:
            break
    
    X0 = torch.stack(X0)
    X1 = torch.stack(X1)
    
    return train_loader, test_loader, X0, X1

def load_fashion_mnist(batch_size=100, train_ratio=0.8, flatten=True, pca_components=None, binary_classes=(0, 1), seed=42):
    """
    Load the Fashion-MNIST dataset and prepare it for Neural ODE classification.
    
    Parameters:
    - batch_size (int): Size of batches for dataloader
    - train_ratio (float): Ratio of data to use for training (remaining for testing)
    - flatten (bool): Whether to flatten the images or keep 2D structure
    - pca_components (int, optional): If provided, reduce dimensions using PCA
    - binary_classes (tuple): Two class indices to use for binary classification
    - seed (int): Random seed for reproducibility
    
    Returns:
    - train_loader (DataLoader): Training data loader
    - test_loader (DataLoader): Test data loader
    - X0 (Tensor): Features for class 0
    - X1 (Tensor): Features for class 1
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # Download and load Fashion-MNIST dataset
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create full dataset by combining train and test
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    
    # Extract tensors and labels
    all_tensors = []
    all_labels = []
    
    for img, label in full_dataset:
        if flatten:
            # Flatten the image
            img = img.view(-1)
        
        # Filter for binary classification (keep only the specified classes)
        if label in binary_classes:
            all_tensors.append(img)
            # Convert to binary label (0 or 1)
            binary_label = 0 if label == binary_classes[0] else 1
            all_labels.append(binary_label)
    
    # Convert to tensors
    X = torch.stack(all_tensors)
    y = torch.tensor(all_labels)
    
    # Apply PCA if requested
    if pca_components is not None and pca_components < X.shape[1]:
        X_np = X.numpy()
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X_np)
        X = torch.tensor(X_reduced, dtype=torch.float32)
        print(f"Reduced dimensions from {X_np.shape[1]} to {pca_components} using PCA")
        print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Split into train and test
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract samples for each class for visualization
    X0 = []
    X1 = []
    
    for features, label in dataset:
        if label == 0 and len(X0) < 500:  # Limit to 500 samples per class
            X0.append(features)
        elif label == 1 and len(X1) < 500:
            X1.append(features)
        
        # Break if we have enough samples
        if len(X0) >= 500 and len(X1) >= 500:
            break
    
    X0 = torch.stack(X0)
    X1 = torch.stack(X1)
    
    return train_loader, test_loader, X0, X1

def project_to_2d(X, method='pca', perplexity=30, n_iter=1000, seed=42):
    """
    Project high-dimensional data to 2D for visualization.
    
    Parameters:
    - X (Tensor): Input data with shape (n_samples, n_features)
    - method (str): Projection method ('pca' or 'tsne')
    - perplexity (int): Perplexity parameter for t-SNE
    - n_iter (int): Number of iterations for t-SNE
    - seed (int): Random seed for reproducibility
    
    Returns:
    - X_2d (ndarray): Projected 2D data with shape (n_samples, 2)
    """
    from sklearn.decomposition import PCA
    
    # Convert tensor to numpy array
    if isinstance(X, torch.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
    
    # Apply projection method
    if method.lower() == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=2, random_state=seed)
        X_2d = pca.fit_transform(X_np)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    elif method.lower() == 'tsne':
        # t-Distributed Stochastic Neighbor Embedding
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=seed)
        X_2d = tsne.fit_transform(X_np)
    
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    return X_2d