"""
plots.py - Visualization Module for Neural ODEs
----------------------------------------------
This module provides functions for visualizing the behavior of Neural ODEs:
- levelsets: Plot decision boundaries and level sets
- loss_evolution: Visualize loss during training
- plot_data: Visualize data points and their trajectories

Workflow position: Analysis & Visualization - helps understand model performance and behavior
"""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter

@torch.no_grad()
def levelsets(model, ax=None, path=None, fig_name=None, footnote=None, 
              contour=True, bar=True, plotlim=[-0.5, 1.5], step=0.01, dpi=100, 
              points=[], transformed_sets=False, return_fig=False):
    """
    Generates and plots the level sets (decision boundaries) for a given model.
    
    Parameters:
    - model (nn.Module): The model whose level sets are to be plotted.
    - ax (matplotlib.axes.Axes): Axes on which to plot. If None, a new figure and axes are created.
    - path (str): Path where the plot should be saved.
    - fig_name (str): Name of the figure file.
    - footnote (str): Footnote text to be added to the figure.
    - contour (bool): Whether to plot contours.
    - bar (bool): Whether to add a color bar.
    - plotlim (list): Plot limits for the x and y axes.
    - step (float): Step size for the meshgrid.
    - dpi (int): Dots per inch for the figure.
    - points (list): List containing X0 and X1 points to be plotted.
    - transformed_sets (bool): Whether to plot transformed sets.
    - return_fig (bool): Whether to return the figure and axes.

    Returns:
    - If return_fig is True, returns the figure and axes.
    """
    # Determine device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Unpack data points for visualization
    X0, X1 = points

    # Initialize figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    
    # Set axis labels
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    
    # Add footnote if provided
    if footnote:
        plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)

    # Move model to the appropriate device
    model.to(device)

    # Generate a meshgrid for the model input
    x1 = torch.arange(plotlim[0], plotlim[1], step, device=device)
    x2 = torch.arange(plotlim[0], plotlim[1], step, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='xy')
    model_inputs = torch.stack([xx1, xx2], dim=-1)

    # Get model predictions on the meshgrid
    if model.trained:
        # If model is trained, use the best parameters
        preds, _ = model(model_inputs, 0)
    else:
        # Otherwise use current parameters
        preds, _ = model(model_inputs)

    # Handle classification outputs with multiple dimensions
    if model.output_dim > 1:
        # Apply softmax to get probabilities
        preds = torch.nn.Softmax(dim=2)(preds)

    # Adjust coordinates if using transformed sets
    if transformed_sets:
        # Use transformed coordinates instead of original
        transformed_inputs, _ = model(model_inputs, 0)
        xx1, xx2 = transformed_inputs[:, :, 0], transformed_inputs[:, :, 1]

    # Prepare prediction data for plotting
    if preds.dim() == 3:
        # For multi-dimensional predictions, use the first class probability
        preds = preds[:, :, 0]  # Assuming we are interested in class 1 probability
        preds = preds.unsqueeze(2)

    # Set plot parameters
    ax.set_xlim(plotlim)
    ax.set_ylim(plotlim)
    ax.set_aspect('equal')
    ax.grid(False)
        
    # Plot the contours if required
    if contour:
        if preds.dim() == 3:
            # Create a colormap gradient from orange to white to blue
            colors = ['#FF5733', [1, 1, 1], to_rgb("C0")]  # Orange to White to Blue
            cm = LinearSegmentedColormap.from_list("Custom", colors, N=40)
            z = preds.detach().cpu().numpy().reshape(xx1.shape)
            levels = np.linspace(0., 1., 8).tolist()
        elif preds.dim() == 2:
            # Alternative colormap for 2D predictions
            colors = [to_rgb("C0"), [1, 1, 1], '#FF5733']  # Blue to White to Orange
            cm = LinearSegmentedColormap.from_list("Custom", colors, N=40)
            z = preds.detach().cpu().numpy()
            z = np.clip(z, 0, 1)  # Ensure values are in [0,1]
            levels = np.linspace(0, 1., 8).tolist()

        # Convert tensors to numpy for plotting
        xx1_np = xx1.detach().cpu().numpy()
        xx2_np = xx2.detach().cpu().numpy()
        
        # Create filled contour plot
        cont = ax.contourf(xx1_np, xx2_np, z, levels, alpha=1, cmap=cm, zorder=0)
        
        # Add colorbar if requested
        if bar:
            cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Prediction Probability')

    # Plot data points with appropriate markers and colors
    point_size = 16
    ax.scatter(X0[:, 0], X0[:, 1], c='C0', marker='X', edgecolor="black", 
               linewidth=0.65, alpha=0.75, s=point_size, label='Class 0')
    ax.scatter(X1[:, 0], X1[:, 1], c='#FF5733', marker='o', edgecolor="black", 
               linewidth=0.65, alpha=0.75, s=point_size, label='Class 1')
    
    # Maintain equal aspect ratio
    ax.set_aspect('equal')

    # Save the figure if requested
    if fig_name and path:
        full_path = os.path.join(path if path else '', fig_name + '.png')
        plt.savefig(full_path, bbox_inches='tight', dpi=400, format='png', facecolor='white')
        print(f"Saved plot to {full_path}")  # Add this line to confirm saving
        if not return_fig:
            plt.clf()
            plt.close(fig)

    # Return figure and axes if requested
    if return_fig:
        return fig, ax

def loss_evolution(trainer, epoch, path, filename='', figsize=(8, 6), footnote=None, 
                  showl2norm=False):
    """
    Plots the evolution of the loss during training.

    Parameters:
    - trainer (object): The trainer object containing the history of training losses.
    - epoch (int): The current epoch number.
    - path (str): Path where the plot should be saved.
    - filename (str): Name of the file where the plot should be saved.
    - figsize (tuple): Size of the figure.
    - footnote (str): Footnote text to be added to the figure.
    - showl2norm (bool): Whether to show L2 norm of the parameters.

    Returns:
    - None
    """
    # Create figure and axes
    fig, ax = plt.subplots(dpi=100, figsize=figsize)
    labelsize = 10

    # Create scale for x-axis (epochs)
    epoch_scale = list(range(1, len(trainer.histories['loss_history']) + 1))
    
    # Plot full loss history with transparency
    ax.plot(epoch_scale, trainer.histories['loss_history'], 'k', alpha=0.5, label='Training Loss')
    
    # Highlight the plot up to the current epoch
    ax.plot(epoch_scale[:epoch], trainer.histories['loss_history'][:epoch], color='k')
    
    # Mark the current epoch with a point
    ax.scatter(epoch + 1, trainer.histories['loss_history'][epoch], color='k', zorder=1)

    # Optionally plot L2 norm of parameters
    if showl2norm:
        ax.plot(epoch_scale, trainer.histories['l2normparam_history'], 'C3--', 
                alpha=0.5, label='L2 Norm of Parameters')
        ax.plot(epoch_scale[:epoch], trainer.histories['l2normparam_history'][:epoch], 'C3--')
        ax.scatter(epoch + 1, trainer.histories['l2normparam_history'][epoch], color='C3', zorder=1)

    # Set plot appearance
    ax.autoscale_view()
    ax.grid(True, zorder=-2)
    ax.yaxis.tick_right()
    ax.set_aspect('auto')
    ax.set_axisbelow(True)
    ax.set_xlabel('Epochs', size=labelsize)
    ax.set_ylabel('Loss', size=labelsize)
    ax.legend(prop={'size': labelsize}, framealpha=1)

    # Add footnote if provided
    if footnote:
        plt.figtext(0.5, -0.005, footnote, ha="center", fontsize=9)

    # Save or display the figure
    if filename:
        figname = os.path.join(path, filename)
        plt.savefig(figname, bbox_inches='tight', dpi=100, format='png', facecolor='white')
        plt.close(fig)
    else:
        plt.show()

def plot_data(model, inputs, targets, N, dpi=200, alpha=0.75, path='', 
              trajs=False, init=False, final=False, rescale=False):
    """
    Plots data points or their trajectories as generated by the model.

    Parameters:
    - model (torch.nn.Module): The model used to generate trajectories.
    - inputs (Tensor): Input data points.
    - targets (Tensor): Labels of the data points.
    - N (int): Number of data points.
    - dpi (int): Resolution of the plot.
    - alpha (float): Transparency level of the plotted points.
    - path (str): Directory where the plot will be saved.
    - trajs (bool): If True, plot trajectories of the data points.
    - init (bool): If True, plot initial points.
    - final (bool): If True, plot final points.
    - rescale (bool): If True, rescale the plot axes.

    Returns:
    - final_image_filename (str): The path to the saved plot image.
    """
    # Extract model parameters
    T, dt = model.T, model.dt
    timesteps = int(T / dt) + 1

    # Configure plotting style
    plt.rcParams.update({
        'xtick.labelsize': 13, 
        'ytick.labelsize': 13,
        'text.usetex': True, 
        'font.family': 'serif',
        'grid.linestyle': 'dotted', 
        'grid.color': 'lightgray'
    })

    # Compute trajectories by running the model forward
    with torch.no_grad():
        _, trajectories = model(inputs)
        trajectories = trajectories.detach().numpy()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 4))  # Fixed image size
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_facecolor('whitesmoke')

    # Helper function to get coordinates at a specific time index
    def get_coords(time_index):
        x_coords = trajectories[time_index, :, 0]
        y_coords = trajectories[time_index, :, -1]
        return x_coords, y_coords

    # Set default output filename
    final_image_filename = os.path.join(path, "plot.png")

    # Handle different plot types
    if trajs:
        # Plot full trajectories for each point
        final_image_filename = os.path.join(path, "trajectories.png")
        for i in range(N):
            ax.plot(
                trajectories[:, i, 0],
                trajectories[:, i, -1],
                color='C0' if targets[i] == 0 else '#FF5733',
                alpha=alpha * 0.5
            )
        x_coords, y_coords = trajectories[:, :, 0], trajectories[:, :, -1]
    else:
        # Plot points at initial or final state
        time_index = 0 if init else timesteps - 1
        x_coords, y_coords = get_coords(time_index)
        final_image_filename = os.path.join(path, "initial_points.png") if init else os.path.join(path, "final_points.png")
        
        # Scatter plot with different markers for different classes
        ax.scatter(
            x_coords[targets == 0], y_coords[targets == 0], 
            c='C0', s=int(3000 / N), alpha=alpha,
            marker='X', linewidth=0.65, edgecolors='black', zorder=3
        )
        ax.scatter(
            x_coords[targets == 1], y_coords[targets == 1], 
            c='#FF5733', s=int(3000 / N), alpha=alpha,
            marker='o', linewidth=0.65, edgecolors='black', zorder=3
        )

    # Determine plot limits and adjust to make it square
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    margin = 0.1
    max_range = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2

    # Adjust limits to the nearest multiples of 0.5 for better readability
    x_min = np.floor((x_center - max_range / 2 - margin * max_range) * 2) / 2
    x_max = np.ceil((x_center + max_range / 2 + margin * max_range) * 2) / 2
    y_min = np.floor((y_center - max_range / 2 - margin * max_range) * 2) / 2
    y_max = np.ceil((y_center + max_range / 2 + margin * max_range) * 2) / 2
    
    # Set axis limits
    if init:
        # For initial points, use fixed or rescaled limits
        if rescale:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        else:
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
    else:
        # For other plots, use dynamically calculated limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Set nicely formatted tick marks
        x_ticks = np.linspace(x_min, x_max, 5)
        y_ticks = np.linspace(y_min, y_max, 5)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Ensure square aspect ratio
    ax.set_aspect('equal')

    # Overlay decision boundary for final plot
    if final:
        # Extract parameters from the model's final layer
        weights = model.linear_layer.weight.data.numpy()
        bias = model.linear_layer.bias.data.numpy()
        
        # Calculate decision boundary
        x_points = np.linspace(x_min, x_max, 2)
        y_points = -weights[0][0] / weights[0][-1] * x_points - (bias[0] - 0.5) / weights[0][-1]
        zz = weights[0][0] * x_points + weights[0][-1] * y_points + bias[0] - 0.5

        # Fill regions based on the decision boundary
        plt.fill_between(x_points, y_points, y_max, color='#F7ABAB', 
                        where=zz >= 0, interpolate=True, alpha=0.5)  # Region for class 1
        plt.fill_between(x_points, y_points, y_min, color='lightblue', 
                        where=zz <= 0, interpolate=True, alpha=0.5)  # Region for class 0
        
        # Plot the decision boundary line
        plt.plot(x_points, y_points, 'k-', lw=2)

    # Save the plot
    plt.savefig(final_image_filename, format='png', dpi=dpi)
    plt.close(fig)
    
    return final_image_filename

def plot_logloss(trainer, full_path, export_fig=False):
    """
    Plots the logarithmic loss over epochs during training.

    Parameters:
    - trainer (object): The trainer object containing the history of training losses.
    - full_path (str): Directory where the plot will be saved.
    - export_fig (bool): If True, save the figure to the specified path.

    Returns:
    - None
    """
    # Convert loss values to logarithmic scale
    log_histories = np.log10(trainer.histories['loss_history'])
    
    # Create figure
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Determine title based on training outcome
    if trainer.classif:
        title = 'Successful classification'
    else:
        if trainer.noimp:
            title = f"Stopping criterion - No improvement for {trainer.patience} epochs"
        elif trainer.relerr:
            title = "Stopping criterion - Relative error"
        elif trainer.nonconv:
            title = "Stopping criterion - Error over threshold in 20000 or 40000 epochs"
        else:
            title = f"Reached maximum number of {trainer.max_epochs} epochs"
            
    title = 'log Error vs Training Epochs: ' + title
    plt.title(title)
    
    # Plot logarithmic loss
    plt.plot(log_histories, color='b', linewidth=2)
    plt.xlabel('Epochs')
    plt.xlim(0, len(trainer.histories['loss_history']) - 1)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure if requested
    if export_fig:
        pathfigslosses = os.path.join(full_path, 'LossVsEpochs.png')
        plt.savefig(pathfigslosses, bbox_inches='tight', dpi=300, format='png', facecolor='white')
    
    plt.show()

# Enhanced visualization functions

def plot_vector_field(model, ax=None, x_min=-2, x_max=2, y_min=-2, y_max=2, t=0.0, 
                     resolution=20, cmap='viridis', density=1.0, with_streamplot=True):
    """
    Plot the vector field (dynamics) of the Neural ODE at a given time.
    
    Parameters:
    - model (nn.Module): Neural ODE model
    - ax (matplotlib.axes): Axes to plot on (optional)
    - x_min, x_max, y_min, y_max (float): Plot boundaries
    - t (float): Time point to evaluate the dynamics
    - resolution (int): Grid resolution
    - cmap (str): Colormap for the vector field
    - density (float): Density of the arrows/streamlines
    - with_streamplot (bool): Whether to use streamplot instead of quiver
    
    Returns:
    - ax (matplotlib.axes): The axes with the plot
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a mesh grid for the vector field
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Prepare input points for the model
    grid_points = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), dtype=torch.float32)
    
    # Compute the vector field using the model dynamics
    with torch.no_grad():
        vector_field = model.fwd_dynamics(torch.tensor(t), grid_points).detach().numpy()
    
    # Reshape the vector field components
    U = vector_field[:, 0].reshape(X.shape)
    V = vector_field[:, 1].reshape(X.shape)
    
    # Compute vector magnitudes for color mapping
    magnitude = np.sqrt(U**2 + V**2)
    
    # Plot the vector field
    if with_streamplot:
        # Use streamplot for a more continuous visualization
        strm = ax.streamplot(X, Y, U, V, color=magnitude, cmap=cmap, 
                            density=density, linewidth=1.5, arrowsize=1.5)
        plt.colorbar(strm.lines, ax=ax, label='Vector magnitude')
    else:
        # Use quiver for a discrete arrow visualization
        q = ax.quiver(X, Y, U, V, magnitude, cmap=cmap, pivot='mid', 
                     width=0.002, scale=50)
        plt.colorbar(q, ax=ax, label='Vector magnitude')
    
    # Set labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Vector Field at t={t:.2f}')
    ax.set_aspect('equal')
    
    return ax

def plot_phase_portrait(model, trajectory, targets, ax=None, 
                      plot_nullclines=True, plot_vector_field_flag=True):  # Cambié el nombre del parámetro
    """
    Plot a phase portrait of the dynamics with an example trajectory.
    
    Parameters:
    - model (nn.Module): Neural ODE model
    - trajectory (Tensor): A pre-computed trajectory through the model
    - targets (Tensor): Labels for coloring the trajectory
    - ax (matplotlib.axes): Axes to plot on (optional)
    - plot_nullclines (bool): Whether to plot nullclines
    - plot_vector_field_flag (bool): Whether to overlay a vector field
    
    Returns:
    - ax (matplotlib.axes): The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract trajectory coordinates
    trajectory = trajectory.detach().numpy()
    x_traj = trajectory[:, :, 0]
    y_traj = trajectory[:, :, 1]
    
    # Determine plot limits from trajectory
    x_min, x_max = np.min(x_traj) - 0.5, np.max(x_traj) + 0.5
    y_min, y_max = np.min(y_traj) - 0.5, np.max(y_traj) + 0.5
    
    # Plot vector field if requested
    if plot_vector_field_flag:  
        plot_vector_field(model, ax, x_min, x_max, y_min, y_max, 
                        t=0.5*model.T, resolution=15, with_streamplot=True)
    
    # Plot nullclines if requested (curves where dx/dt=0 or dy/dt=0)
    if plot_nullclines:
        # Create a fine grid for nullcline detection
        res = 100
        x = np.linspace(x_min, x_max, res)
        y = np.linspace(y_min, y_max, res)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), dtype=torch.float32)
        
        # Compute vector field at middle of time interval
        with torch.no_grad():
            vector_field = model.fwd_dynamics(torch.tensor(0.5*model.T), grid_points).detach().numpy()
        
        # Reshape vector field components
        U = vector_field[:, 0].reshape(X.shape)
        V = vector_field[:, 1].reshape(X.shape)
        
        # Plot dx/dt=0 nullcline (where U is close to zero)
        ax.contour(X, Y, U, levels=[0], colors='blue', linestyles='--', linewidths=2)
        
        # Plot dy/dt=0 nullcline (where V is close to zero)
        ax.contour(X, Y, V, levels=[0], colors='red', linestyles='--', linewidths=2)
    
    # Plot trajectories with colors based on class
    for i in range(trajectory.shape[1]):
        ax.plot(x_traj[:, i], y_traj[:, i], 
               color='C0' if targets[i] <= 0.5 else '#FF5733', 
               alpha=0.7, linewidth=1.5)
        # Mark start and end points
        ax.scatter(x_traj[0, i], y_traj[0, i], color='green', zorder=5, s=50, label='Start' if i==0 else "")
        ax.scatter(x_traj[-1, i], y_traj[-1, i], color='red', zorder=5, s=50, label='End' if i==0 else "")
    
    # Add labels and legend
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Phase Portrait with Trajectories')
    ax.legend()
    ax.set_aspect('equal')
    
    return ax

def visualize_model_comparison(models, dataset, path=None, filename=None):
    """
    Create a comparison visualization for multiple models.
    
    Parameters:
    - models (dict): Dictionary of {model_name: model}
    - dataset (DataLoader): Dataset for visualization
    - path (str): Directory to save the visualization
    - filename (str): Filename for saving
    
    Returns:
    - fig (matplotlib.figure): The comparison figure
    """
    # Get sample data
    inputs, targets = next(iter(dataset))
    
    # Determine number of models
    n_models = len(models)
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5), dpi=150)
    
    # Handle case with single model
    if n_models == 1:
        axes = [axes]
    
    # Plot decision boundary for each model
    for ax, (model_name, model) in zip(axes, models.items()):
        # Run predictions
        with torch.no_grad():
            pred, trajectory = model(inputs)
        
        # Split data by class
        mask_0 = targets == 0
        mask_1 = targets == 1
        X0, X1 = inputs[mask_0], inputs[mask_1]
        
        # Plot decision boundary
        levelsets(model, ax=ax, points=[X0, X1], contour=True, bar=False)
        
        # Set title
        ax.set_title(model_name)
    
    # Add overall title
    fig.suptitle('Model Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save if path is provided
    if path and filename:
        full_path = os.path.join(path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    return fig