"""
gifs.py - Trajectory Visualization Module
----------------------------------------
This module provides functions to visualize the trajectories of data points
as they flow through the Neural ODE model. The main functionality includes:
- Creating GIF animations of trajectories
- Plotting hyperplanes and decision boundaries
- Selecting random samples for visualization

Workflow position: Visualization - helps understand model behavior and results
"""

import os
import imageio
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import rc    
# rc for runtime configuration, interp1d for interpolation

def traj_gif(model, inputs, targets, dpi=200, path='', fps=5):
    """
    Generate an animated GIF showing how data points flow through the model over time.
    
    Parameters:
    - model (nn.Module): Trained Neural ODE model
    - inputs (Tensor): Input data points
    - targets (Tensor): Labels for the data points
    - dpi (int): Resolution of the output images
    - path (str): Directory to save the output files
    - fps (int): Frames per second for the GIF animation
    
    Returns:
    - str: Path to the created GIF file
    """
    # Plotting configuration
    rc("text", usetex=True)
    plt.rcParams.update({'font.size': 18, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 
                         'text.usetex': True, 'font.family': 'serif',
                         'grid.linestyle': 'dotted', 'grid.color': 'lightgray'})
    alpha, alpha_line = 0.9, 0.5

    # Visualization options
    dyn_lims = False  # If True, the limits of the plot are dynamic
    normalize = False  # If True, the plot is normalized
    hyp = True        # If True, the hyperplanes are plotted
    paths = False     # If True, paths are plotted
        
    # Validate filename
    filename = f"trajectories.gif"
    if not filename.endswith(".gif"):
        raise RuntimeError(f"Name must end with .gif, but ends with {filename}")
    base_filename = filename[:-4]
    
    # Define colors based on targets (blue for class 1, orange for class 0)
    color = ['C0' if t > 0.5 else '#FF5733' for t in targets]
    
    # Compute trajectories through the model
    _, trajectories = model(inputs)
    trajectories = trajectories.detach().numpy()
    
    # Normalize trajectories if required
    if normalize:
        # Apply tanh to constrain values between -1 and 1
        trajectories = np.tanh(trajectories)
        x_min, x_max, y_min, y_max = -1.1, 1.1, -1.1, 1.1
    else:
        # Calculate plot limits based on trajectory bounds
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
        # Add margins for better visualization
        margin = 0.1
        x_range, y_range = x_max - x_min, y_max - y_min
        x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
        y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
    
    # Calculate time steps for trajectory interpolation
    T, dt = model.T, model.dt
    timesteps = int(T / dt) + 1
    integration_time = torch.linspace(0.0, T, timesteps).numpy()
    
    # Interpolate trajectories for smoother animation
    interp_time = 120  # Number of frames in the animation
    _time = torch.linspace(0.0, T, interp_time).numpy()
    
    # Create interpolation functions for each trajectory coordinate
    interp_funcs = [interp1d(integration_time, trajectories[:, i, j], kind='cubic', fill_value='extrapolate') 
                    for i in range(inputs.shape[0]) for j in range(2)]
    
    # Generate frames for GIF
    gif_names = []
    for t in range(interp_time):
        # Create figure for the current time frame
        fig, ax = plt.subplots()
        current_time = _time[t]
        title = f"$N={len(targets)}$, $t={current_time:.2f}$"
        plt.title(title, fontsize=20)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_facecolor('whitesmoke')
        
        # Calculate current point coordinates at this time step
        x_coords = [func(current_time) for func in interp_funcs[::2]]
        y_coords = [func(current_time) for func in interp_funcs[1::2]]

        # Update dynamic limits if enabled
        if not normalize and dyn_lims:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range, y_range = x_max - x_min, y_max - y_min
            x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
            y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Plot points at current time step
        ax.scatter(x_coords, y_coords, c=color, alpha=alpha, marker='o', 
                  linewidth=0.65, edgecolors='black', zorder=3)
        
        # Calculate time step index for model parameters
        k = int(_time[t] * model.num_vals / T)
        k = min(k, model.num_vals - 1)  # Ensure index is within bounds
        
        # Optionally plot hyperplanes and trajectories
        if hyp and t < interp_time - 1:
            _plot_hyperplanes(ax, model, x_min, x_max, y_min, y_max, k)
        if t == interp_time - 1:
            _plot_boundary(ax, model, x_min, x_max, y_min, y_max)
        if t > 0 and paths:
            _plot_paths(ax, interp_funcs, _time, t, color, alpha_line, inputs.shape[0])

        # Save frame as image
        frame_filename = os.path.join(path, f"{base_filename}_{t}.png")
        gif_names.append(frame_filename)
        plt.savefig(frame_filename, format='png', dpi=dpi)
        plt.close(fig)
    
    # Create GIF from saved frames
    gif_path = _create_gif(gif_names, path, filename, fps)
    return gif_path

def _plot_hyperplanes(ax, model, x_min, x_max, y_min, y_max, k):
    """
    Plot the hyperplanes (decision boundaries) for a specific time step.
    
    Parameters:
    - ax (matplotlib.axes): Axes to plot on
    - model (nn.Module): Neural ODE model
    - x_min, x_max, y_min, y_max (float): Plot limits
    - k (int): Time step index
    
    Returns:
    - None: Updates the provided axes
    """
    # Get weights from the model's final layer
    weights = model.linear_layer.weight.data.numpy()
    
    # Create a linear space for plotting
    x_points = np.linspace(x_min, x_max, 100)
    
    # Plot decision boundaries based on model architecture
    for i in range(weights.shape[0]):
        # Get time-dependent weights and biases based on architecture
        if model.architecture == 'bottleneck':
            a = model.fwd_dynamics.fc1_time[k].weight.detach().numpy()
            b = model.fwd_dynamics.fc1_time[k].bias.detach().numpy()
        elif model.architecture == 'inside':
            a = model.fwd_dynamics.fc2_time[k].weight.detach().numpy()
            b = model.fwd_dynamics.fc2_time[k].bias.detach().numpy()
        else:  # 'outside' architecture
            b = model.fwd_dynamics.fc2_time[k].bias.detach().numpy()
            a = model.fwd_dynamics.fc2_time[k].weight.detach().numpy()
        
        # Create meshgrid for contour plot
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), 
                             np.linspace(y_min, y_max, 500))
        # Compute hyperplane values at each point
        zz = a[i][0] * xx + a[i][1] * yy + b[i]
        
        # Plot filled contours for the hyperplanes
        ax.contourf(xx, yy, zz, levels=[-np.inf, 0], colors='black', alpha=0.75)
        ax.contourf(xx, yy, zz, levels=[0, np.inf], colors='lightgray', alpha=0.25)
    
        # Plot hyperplane lines
        if weights[i][1] == 0:  # Vertical line case
            y_points = np.linspace(y_min, y_max, 100)
            x_points = -b[i] / a[i][0]
        else:  # General line case
            x_points = np.linspace(x_min, x_max, 100)
            y_points = -a[i][0] / a[i][1] * x_points - b[i] / a[i][1]
        ax.plot(x_points, y_points, 'k--', lw=2)

def _plot_boundary(ax, model, x_min, x_max, y_min, y_max):
    """
    Plot the final decision boundary based on the model's linear layer.
    
    Parameters:
    - ax (matplotlib.axes): Axes to plot on
    - model (nn.Module): Neural ODE model
    - x_min, x_max, y_min, y_max (float): Plot limits
    
    Returns:
    - None: Updates the provided axes
    """
    # Get weights and bias from the final linear layer
    weights = model.linear_layer.weight.data.numpy()
    bias = model.linear_layer.bias.data.numpy()
    
    # Sample points for plotting
    x_points = np.linspace(x_min, x_max, 100)

    # Handle special case of vertical decision boundary
    if weights[0][1] == 0:  # Vertical line case
        # Calculate x-intercept: w_1 * x + b = 0.5 => x = (0.5 - b) / w_1
        x_val = - (bias[0] - 0.5) / weights[0][0]
        y_points = np.linspace(y_min, y_max, 100)
        
        # Fill regions on either side of the boundary
        ax.fill_betweenx(y_points, x_val, x_max, color='lightblue', alpha=0.5)  # Region for class 1
        ax.fill_betweenx(y_points, x_val, x_min, color='#F0B27A', alpha=0.5)    # Region for class 0
        
        # Plot the decision boundary
        ax.plot([x_val] * 100, y_points, 'k-', lw=2)
    else:  # Non-vertical line case
        # Calculate line equation: w_1 * x + w_2 * y + b = 0.5 => y = -(w_1 * x + b - 0.5) / w_2
        y_points = -weights[0][0] / weights[0][1] * x_points - (bias[0] - 0.5) / weights[0][1]
        
        # Calculate decision function values for coloring
        zz = weights[0][0] * x_points + weights[0][1] * y_points + bias[0] - 0.5
        
        # Fill regions based on decision function
        ax.fill_between(x_points, y_points, y_max, color='lightblue', 
                        where=zz >= 0, interpolate=True, alpha=0.5)  # Region for class 1
        ax.fill_between(x_points, y_points, y_min, color='#F0B27A', 
                        where=zz <= 0, interpolate=True, alpha=0.5)  # Region for class 0
        
        # Plot the decision boundary
        ax.plot(x_points, y_points, 'k-', lw=2)

def _plot_paths(ax, interp_funcs, _time, t, color, alpha_line, num_inputs):
    """
    Plot the path of each point up to the current time.
    
    Parameters:
    - ax (matplotlib.axes): Axes to plot on
    - interp_funcs (list): List of interpolation functions for each coordinate
    - _time (ndarray): Array of time points
    - t (int): Current time index
    - color (list): List of colors for each point
    - alpha_line (float): Transparency for path lines
    - num_inputs (int): Number of input points
    
    Returns:
    - None: Updates the provided axes
    """
    # Plot path for each point
    for i in range(num_inputs):
        # Get x and y coordinates over time
        x_path = interp_funcs[2 * i](_time)[:t + 1]
        y_path = interp_funcs[2 * i + 1](_time)[:t + 1]
        
        # Plot path with appropriate color and transparency
        ax.plot(x_path, y_path, c=color[i], alpha=alpha_line, linewidth=0.75, zorder=1)

def _create_gif(gif_names, path, filename, fps):
    """
    Create a GIF animation from a sequence of images.
    
    Parameters:
    - gif_names (list): List of image file paths
    - path (str): Directory to save the GIF
    - filename (str): Name of the output GIF file
    - fps (int): Frames per second for the animation
    
    Returns:
    - str: Path to the created GIF file
    """
    # Read image files
    imgs = [np.array(imageio.imread(name)) for name in gif_names]
    
    # Create output path
    gif_path = os.path.join(path, filename)
    
    # Write GIF file
    imageio.mimwrite(gif_path, imgs, fps=fps)
    
    # Clean up temporary image files
    for img_path in gif_names:
        os.remove(img_path)
    
    return gif_path

def select_random_samples(data_loader, num_samples):
    """
    Randomly select a subset of samples from a data loader.
    
    Parameters:
    - data_loader (DataLoader): The data loader to sample from
    - num_samples (int): Number of samples to select
    
    Returns:
    - Tensor: Selected input samples
    - Tensor: Corresponding target labels
    """
    # Get total number of samples in the dataset
    total_samples = len(data_loader.dataset)
    
    # Check if requested number is valid
    if num_samples > total_samples:
        raise ValueError("Requested more samples than available in the dataset")
    
    # Randomly select indices from the dataset
    selected_indices = random.sample(range(total_samples), num_samples)

    # Collect the selected samples and their targets
    inputs = torch.stack([data_loader.dataset[i][0] for i in selected_indices])
    targets = torch.stack([data_loader.dataset[i][1] for i in selected_indices])

    return inputs, targets

# Enhanced visualization functions

def create_flow_field(model, x_min, x_max, y_min, y_max, t=0.0, resolution=20, ax=None, cmap='coolwarm'):
    """
    Create a vector field visualization of the ODE dynamics.
    
    Parameters:
    - model (nn.Module): Neural ODE model
    - x_min, x_max, y_min, y_max (float): Plot limits
    - t (float): Time point for evaluation
    - resolution (int): Number of grid points in each dimension
    - ax (matplotlib.axes): Axes to plot on (optional)
    - cmap (str): Colormap for the vector field
    
    Returns:
    - matplotlib.axes: Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create input grid for the model
    grid_shape = X.shape
    grid = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), dtype=torch.float32)
    
    # Compute vector field at time t
    with torch.no_grad():
        vector_field = model.fwd_dynamics(t, grid).detach().numpy()
    
    # Reshape vector field components
    U = vector_field[:, 0].reshape(grid_shape)
    V = vector_field[:, 1].reshape(grid_shape)
    
    # Compute vector magnitudes for color mapping
    magnitude = np.sqrt(U**2 + V**2)
    
    # Plot vector field
    ax.streamplot(X, Y, U, V, density=1.0, linewidth=1.0, arrowsize=1.5, color=magnitude, cmap=cmap)
    
    # Add colorbar
    plt.colorbar(ax.collections[0], ax=ax, label='Vector magnitude')
    
    # Set labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Vector Field at t={t:.2f}')
    
    return ax

def visualize_energy_landscape(model, x_min, x_max, y_min, y_max, t=0.0, resolution=100, ax=None):
    """
    Visualize the energy landscape of the Neural ODE system.
    
    Parameters:
    - model (nn.Module): Neural ODE model
    - x_min, x_max, y_min, y_max (float): Plot limits
    - t (float): Time point for evaluation
    - resolution (int): Number of grid points in each dimension
    - ax (matplotlib.axes): Axes to plot on (optional)
    
    Returns:
    - matplotlib.axes: Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create input grid for the model
    grid = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), dtype=torch.float32)
    
    # Compute vector field and its magnitude (energy)
    with torch.no_grad():
        vector_field = model.fwd_dynamics(t, grid).detach().numpy()
    
    # Compute energy as the negative of vector magnitude
    energy = -np.sum(vector_field**2, axis=1).reshape(X.shape)
    
    # Plot energy landscape as contour
    contour = ax.contourf(X, Y, energy, 20, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(contour, ax=ax, label='Energy')
    
    # Set labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Energy Landscape at t={t:.2f}')
    
    return ax

def create_trajectory_comparison(model1, model2, inputs, targets, t_max=None, num_steps=100,
                                fig_size=(12, 10), dpi=150, path=None, filename=None):
    """
    Create a comparison of trajectories between two models.
    
    Parameters:
    - model1, model2 (nn.Module): Neural ODE models to compare
    - inputs (Tensor): Input data points
    - targets (Tensor): Labels for the data points
    - t_max (float): Maximum time (if None, uses max of model T values)
    - num_steps (int): Number of time steps for trajectory computation
    - fig_size (tuple): Figure size
    - dpi (int): Resolution
    - path (str): Path to save the figure
    - filename (str): Filename for saving
    
    Returns:
    - matplotlib.figure: Figure with the comparison
    """
    # Determine maximum time
    if t_max is None:
        t_max = max(model1.T, model2.T)
    
    # Create evaluation time points
    times = torch.linspace(0.0, t_max, num_steps)
    
    # Compute trajectories for both models
    with torch.no_grad():
        traj1 = model1.flow.trajectory(inputs, num_steps).detach().numpy()
        traj2 = model2.flow.trajectory(inputs, num_steps).detach().numpy()
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fig_size, dpi=dpi)
    
    # Color map based on targets
    colors = ['C0' if t > 0.5 else '#FF5733' for t in targets]
    
    # Plot initial state (same for both models)
    ax1.scatter(inputs[:, 0], inputs[:, 1], c=colors, edgecolors='black', alpha=0.8)
    ax1.set_title('Initial State')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    
    # Plot final state for model 1
    ax2.scatter(traj1[-1, :, 0], traj1[-1, :, 1], c=colors, edgecolors='black', alpha=0.8)
    ax2.set_title(f'Model 1 Final State (t={t_max:.2f})')
    ax2.set_xlabel('$x_1$')
    
    # Plot final state for model 2
    ax3.scatter(traj2[-1, :, 0], traj2[-1, :, 1], c=colors, edgecolors='black', alpha=0.8)
    ax3.set_title(f'Model 2 Final State (t={t_max:.2f})')
    ax3.set_xlabel('$x_1$')
    
    # Equalize axis limits
    all_axes = [ax1, ax2, ax3]
    x_min = min(ax.get_xlim()[0] for ax in all_axes)
    x_max = max(ax.get_xlim()[1] for ax in all_axes)
    y_min = min(ax.get_ylim()[0] for ax in all_axes)
    y_max = max(ax.get_ylim()[1] for ax in all_axes)
    
    for ax in all_axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Add figure title
    plt.suptitle('Trajectory Comparison Between Models', fontsize=16)
    plt.tight_layout()
    
    # Save figure if path and filename provided
    if path and filename:
        full_path = os.path.join(path, filename)
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    
    return fig