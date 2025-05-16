"""
training.py - Neural ODE Training Module
----------------------------------------
This module handles the training process for Neural ODE models, including:
- Loss functions for different classification tasks
- Training loop with early stopping and model checkpointing
- Optimization and evaluation of model performance

Workflow position: Model training - optimizes model parameters using training data
"""
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np

class CustomLoss(nn.Module):
    """
    Custom margin-based loss function for classification.
    Encourages a margin of separation between classes.
    """
    def __init__(self, margin=0.1):
        super(CustomLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_batch):
        """
        Compute margin-based loss.
        
        Parameters:
        - y_pred (Tensor): Model predictions
        - y_batch (Tensor): Ground truth labels
        
        Returns:
        - Tensor: Mean loss value
        """
        margin = self.margin
        # Convert labels to +1/-1 format for margin calculation
        sign_y = 1 - 2 * y_batch.float()
        # Calculate loss with margin
        loss = F.relu(sign_y * (y_pred - 0.5 + sign_y * margin))
        return loss.mean()

# Dictionary of available loss functions
losses = {
    'mse': nn.MSELoss(),                # Mean squared error (regression)
    'cross_entropy': nn.CrossEntropyLoss(),  # Cross entropy (multi-class)
    'bce': nn.BCEWithLogitsLoss(),      # Binary cross entropy with logits
    'linear_sep': CustomLoss(),         # Custom margin-based loss
}

class Trainer():
    """
    Training loop for Neural ODE models.
    Handles optimization, early stopping, and performance tracking.
    """
    def __init__(self, model, optimizer, device, fixed_projector, loss_func='cross_entropy', AdjTrainer=False,
                 print_freq=10, record_freq=20, pathgifs='', verbose=True):
        """
        Initialize the trainer.
        
        Parameters:
        - model (nn.Module): Neural ODE model
        - optimizer (torch.optim): Optimizer for parameter updates
        - device (torch.device): Device for computation (CPU/GPU)
        - fixed_projector (bool): Whether the final layer is fixed
        - loss_func (str): Name of the loss function to use
        - AdjTrainer (bool): Whether to use adjoint method for training
        - print_freq (int): How often to print progress
        - record_freq (int): How often to record metrics
        - pathgifs (str): Path to save visualizations
        - verbose (bool): Whether to print detailed information
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_func = loss_func
        self.loss = losses[loss_func]
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.pathgifs = pathgifs
        self.verbose = verbose
        self.adjtrainer = AdjTrainer
        self.fixed_projector = fixed_projector
        
    def train(self, datatrain, max_epochs=30000, pathparams='', validation_data=None, 
              patience=10000, lr_scheduler=None):
        """
        Train the model.
        
        Parameters:
        - datatrain (DataLoader): Training data
        - max_epochs (int): Maximum number of epochs to train
        - pathparams (str): Path to save model parameters
        - validation_data (DataLoader): Validation data for early stopping
        - patience (int): Number of epochs to wait for improvement before stopping
        - lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        
        Returns:
        - None: Updates model parameters in-place
        """
        # Initialize flags and parameters
        self.classif = self.noimp = self.relerr = self.nonconv = False
        self.max_epochs = max_epochs
        self.patience = patience if patience is not None else 10000
        self.best_score = float('inf')
        self.best_epoch = 0
        self.model.best_param = []
        last_saved_model = None
        
        # Initialize history tracking
        self.histories = {
            'loss_history': [],
            'acc_history': [],
            'val_loss_history': [],
            'val_acc_history': [],
            'time_per_epoch': []
        }

        # Collect points and labels from datatrain for final classification check
        points, labels = zip(*datatrain)
        points, labels = torch.cat(points), torch.cat(labels)

        # Set relative tolerance and consecutive iteration checks for convergence
        rel_tol, consecutive_iterations = 1e-10, 20
        
        if self.verbose:
            print(f'Training for up to {self.max_epochs} epochs with patience {self.patience}')
            print(f'Optimization method: {self.optimizer.__class__.__name__}')
            print(f'Loss function: {self.loss_func}')
        
        # Main training loop
        from tqdm import tqdm
        with tqdm(total=self.max_epochs, desc="Epochs") as pbar:
            for epoch in range(self.max_epochs):
                # Start timing the epoch
                start_time = time.time()
                
                # Run one epoch of training
                train_loss, train_acc = self.run_epoch(datatrain)
                
                # Calculate epoch time
                epoch_time = time.time() - start_time
                self.histories['time_per_epoch'].append(epoch_time)
                
                # Log training metrics
                self.histories['loss_history'].append(train_loss)
                if not self.fixed_projector and self.loss_func == 'cross_entropy':
                    self.histories['acc_history'].append(train_acc)
                
                # Print progress if enabled
                if self.verbose and (epoch + 1) % self.print_freq == 0:
                    pbar.write(f"Epoch {epoch + 1}/{self.max_epochs} ({epoch_time:.2f}s). "
                        f"Train loss: {train_loss:.6f}" + 
                        (f", Train acc: {train_acc:.4f}" if 'acc_history' in self.histories else ""))
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=f"{train_loss:.6f}")
                
                # Validate if validation data is provided
                if validation_data is not None:
                    val_loss, val_acc = self.evaluate(validation_data)
                    self.histories['val_loss_history'].append(val_loss)
                    self.histories['val_acc_history'].append(val_acc)
                    
                    if self.verbose and (epoch + 1) % self.print_freq == 0:
                        print(f"Validation loss: {val_loss:.6f}, Validation acc: {val_acc:.4f}")
                    
                    # Use validation loss for model selection
                    current_score = val_loss
                else:
                    # Use training loss for model selection if no validation data
                    current_score = train_loss
                
                # Update learning rate if scheduler is provided
                if lr_scheduler is not None:
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(current_score)
                    else:
                        lr_scheduler.step()
                
                # Check if model improved
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_epoch = epoch + 1
                    
                    # Save model state
                    self.model.best_param = self.model.state_dict()
                    
                    # Remove previous saved model if it exists
                    if last_saved_model is not None and os.path.exists(last_saved_model):
                        os.remove(last_saved_model)
                    
                    # Save the current model
                    if pathparams:
                        os.makedirs(pathparams, exist_ok=True)
                        model_filename = os.path.join(pathparams, f'best_param_NODE.pt')
                        torch.save(self.model.state_dict(), model_filename)
                        if self.verbose:
                            print(f'Saving model to {model_filename}')
                        last_saved_model = model_filename
                    
                    # Reset patience counter
                    pat_epochs = 0
                else:
                    # Increment patience counter
                    pat_epochs += 1
                    if pat_epochs >= self.patience:
                        if self.verbose:
                            print(f'Early stopping: No improvement for {self.patience} epochs')
                        self.noimp = True
                        break   
                
                # Check for slow convergence
                if (epoch > 20000 and self.best_score > 0.15) or (epoch > 40000 and self.best_score > 0.1):
                    if self.verbose:
                        print(f'Stopping early due to slow convergence')
                    self.nonconv = True
                    break
                    
                # Check for convergence based on relative error
                if len(self.histories['loss_history']) > consecutive_iterations:
                    # Calculate relative error over last consecutive_iterations
                    rel_error = max(abs(self.histories['loss_history'][-1] - self.histories['loss_history'][-2-j]) / 
                                max(self.histories['loss_history'][-2-j], 1e-10) 
                                for j in range(consecutive_iterations))
                                
                    if rel_error < rel_tol:
                        if self.verbose:
                            print(f'Converged: Relative error {rel_error:.10f} < {rel_tol} for {consecutive_iterations} iterations')
                        self.relerr = True
                        break

                # Check if perfect classification is achieved on training data
                # This is used as an additional stopping criterion
                with torch.no_grad():
                    predictions, _ = self.model(points)
                    if self.loss_func == 'cross_entropy' and self.model.output_dim > 1:
                        predictions = torch.argmax(F.softmax(predictions, dim=1), dim=1)
                        correct = (predictions == labels).all()
                    else:
                        condition_1 = (labels == 1) & (predictions > 0.5)
                        condition_0 = (labels == 0) & (predictions < 0.5)
                        correct = (condition_0 | condition_1).all()
                    
                    if correct:
                        if self.verbose:
                            print('Perfect classification achieved!')
                        self.trained = True
                        self.classif = True
                        self.model.trained = True
                        
                        # Load best parameters
                        if self.model.best_param:
                            self.model.load_state_dict(self.model.best_param)
                        return
        
        # Training loop completed - either by reaching max_epochs or early stopping
        self.trained = True
        self.model.trained = True
        
        # Load best parameters
        if self.model.best_param:
            self.model.load_state_dict(self.model.best_param)
            
        if self.verbose:
            print(f"Training completed. Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
        return
    
    

    # Then modify the run_epoch method in the Trainer class:
    def run_epoch(self, data):
        """
        Run a single training epoch.
        
        Parameters:
        - data (DataLoader): Training data
        
        Returns:
        - float: Average loss for the epoch
        - float: Average accuracy for the epoch (if applicable)
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize variables
        epoch_loss = 0.0
        epoch_acc = 0.0
        data_len = len(data)

        # Loop through batches with progress bar
        for _, (x_batch, y_batch) in enumerate(tqdm(data, desc="Training", leave=False)):
            # Prepare data based on loss function
            if self.loss_func == 'mse' or self.loss_func == 'bce' or self.loss_func == 'linear_sep':
                y_batch = y_batch.float()
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Move data to device
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            y_pred, _ = self.model(x_batch)
            
            # Compute loss
            loss = self.loss(y_pred, y_batch)
            epoch_loss += loss.item()
            
            # Compute accuracy for classification tasks
            if self.loss_func == 'cross_entropy':
                # Apply softmax for multi-class classification
                m = nn.Softmax(dim=1)
                softpred = torch.argmax(m(y_pred), 1)
                accuracy = (softpred == y_batch).sum().item() / y_batch.size(0)
                epoch_acc += accuracy
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
        
        # Return average loss and accuracy
        return epoch_loss / data_len, epoch_acc / data_len
    
    def evaluate(self, data):
        """
        Evaluate the model on validation or test data.
        
        Parameters:
        - data (DataLoader): Validation or test data
        
        Returns:
        - float: Average loss
        - float: Average accuracy (if applicable)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize variables
        val_loss = 0.0
        val_acc = 0.0
        data_len = len(data)
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(data):
                # Prepare data based on loss function
                if self.loss_func == 'mse' or self.loss_func == 'bce' or self.loss_func == 'linear_sep':
                    y_batch = y_batch.float()
                
                # Move data to device
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                y_pred, _ = self.model(x_batch)
                
                # Compute loss
                loss = self.loss(y_pred, y_batch)
                val_loss += loss.item()
                
                # Compute accuracy for classification tasks
                if self.loss_func == 'cross_entropy':
                    m = nn.Softmax(dim=1)
                    softpred = torch.argmax(m(y_pred), 1)
                    accuracy = (softpred == y_batch).sum().item() / y_batch.size(0)
                    val_acc += accuracy
        
        # Return average loss and accuracy
        return val_loss / data_len, val_acc / data_len
    
    def predict(self, data, return_probs=False):
        """
        Make predictions using the trained model.
        
        Parameters:
        - data (DataLoader or Tensor): Input data
        - return_probs (bool): Whether to return probabilities or class predictions
        
        Returns:
        - Tensor: Predictions or probabilities
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Process input based on type
        if isinstance(data, torch.utils.data.DataLoader):
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for x_batch, _ in data:
                    x_batch = x_batch.to(self.device)
                    
                    # Get predictions
                    y_pred, _ = self.model(x_batch)
                    
                    # Process based on loss function
                    if self.loss_func == 'cross_entropy' and self.model.output_dim > 1:
                        probs = F.softmax(y_pred, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        all_probs.append(probs)
                    else:
                        probs = torch.sigmoid(y_pred) if self.loss_func == 'bce' else y_pred
                        preds = (probs > 0.5).float()
                        all_probs.append(probs)
                    
                    all_preds.append(preds)
            
            # Concatenate batch predictions
            all_preds = torch.cat(all_preds)
            all_probs = torch.cat(all_probs)
            
            return all_probs if return_probs else all_preds
        else:
            # Process a single tensor
            with torch.no_grad():
                data = data.to(self.device)
                y_pred, _ = self.model(data)
                
                # Process based on loss function
                if self.loss_func == 'cross_entropy' and self.model.output_dim > 1:
                    probs = F.softmax(y_pred, dim=1)
                    preds = torch.argmax(probs, dim=1)
                else:
                    probs = torch.sigmoid(y_pred) if self.loss_func == 'bce' else y_pred
                    preds = (probs > 0.5).float()
                
                return probs if return_probs else preds

# Additional utility functions for enhanced training capabilities

def create_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0.0):
    """
    Create an optimizer based on name and parameters.
    
    Parameters:
    - model (nn.Module): Model to optimize
    - optimizer_name (str): Name of the optimizer ('adam', 'sgd', 'adamw', etc.)
    - lr (float): Learning rate
    - weight_decay (float): Weight decay for regularization
    
    Returns:
    - optimizer: PyTorch optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

def create_lr_scheduler(optimizer, scheduler_name='plateau', **kwargs):
    """
    Create a learning rate scheduler.
    
    Parameters:
    - optimizer: PyTorch optimizer
    - scheduler_name (str): Name of the scheduler
    - **kwargs: Additional parameters for the scheduler
    
    Returns:
    - scheduler: PyTorch learning rate scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'plateau':
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.5)
        min_lr = kwargs.get('min_lr', 1e-6)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True, min_lr=min_lr
        )
    elif scheduler_name == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")

def compute_metrics(model, data, metric_names=None):
    """
    Compute various evaluation metrics.
    
    Parameters:
    - model (nn.Module): Trained model
    - data (DataLoader): Evaluation data
    - metric_names (list): List of metrics to compute
    
    Returns:
    - dict: Dictionary of computed metrics
    """
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
    
    # Default metrics dictionary
    metrics = {name: 0.0 for name in metric_names}
    
    # Get predictions and true labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in data:
            x_batch = x_batch.to(model.device)
            
            # Get predictions
            y_pred, _ = model(x_batch)
            
            # Convert to binary predictions for binary classification
            if model.output_dim == 1 or (hasattr(model, 'loss_func') and model.loss_func in ['bce', 'linear_sep']):
                preds = (y_pred > 0.5).float()
            else:
                preds = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())
    
    # Concatenate batch results
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    
    # Accuracy
    if 'accuracy' in metric_names:
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Precision
    if 'precision' in metric_names:
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall
    if 'recall' in metric_names:
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    if 'f1' in metric_names:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return metrics