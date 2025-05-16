"""
neural_odes.py - Neural ODE Implementation
------------------------------------------
This module implements Neural Ordinary Differential Equations (Neural ODEs) for classification.
It defines the core components of the model:
- Dynamics: the vector field that defines how the system evolves
- Semiflow: computes the trajectory by integrating the dynamics
- NeuralODE: combines dynamics and semiflow with optional final layer for classification
- adj_Dynamics: implements the adjoint method for efficient backpropagation

Workflow position: Model architecture - defines how data flows through the neural ODE
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

# Set the maximum number of steps for the ODE solver to prevent infinite loops
MAX_NUM_STEPS = 1000

# Define custom activation functions and their derivatives
def tworelu(input):
    """Square of ReLU: (max(0,x))²"""
    return torch.relu(input)**2 

def trunrelu(input):
    """Truncated ReLU: min(max(0,x),1)"""
    return torch.clamp(input, min=0, max=1)

def tanh_prime(input):
    """Derivative of the tanh function: 1 - tanh²(x)"""
    return 1 - torch.tanh(input) ** 2

def relu_prime(input):
    """Derivative of the ReLU function: 1 if x > 0 else 0"""
    return (input >= 0).float()

# Dictionary of activation functions for easy access
activations = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'leakyrelu': nn.LeakyReLU(negative_slope=0.25, inplace=True),
    '2relu': tworelu,
    'trunrelu': trunrelu,
    'tanh_prime': tanh_prime,
    'relu_prime': relu_prime,               
}

# Dictionary mapping architecture types to internal numeric codes
architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1}

class Dynamics(nn.Module):
    """
    Defines the nonlinear dynamics for the neural ODE. 
    This represents the vector field f(t,x) in the ODE: dx/dt = f(t,x)
    
    Supports multiple architectures:
    - 'inside': applies nonlinearity inside (σ(W·x))
    - 'outside': applies nonlinearity outside (W·σ(x))
    - 'bottleneck': uses encoder-decoder structure with nonlinearity in between
    """
    def __init__(self, device, input_dim, hidden_dim, non_linearity='tanh', architecture='inside', 
                 T=10, num_vals=10, seed_params=1):
        super(Dynamics, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed_params)
        torch.cuda.manual_seed_all(seed_params)
        
        self.sigma = non_linearity
        
        # Ensure valid non_linearity and architecture
        if non_linearity not in activations or architecture not in architectures:
            raise ValueError("Invalid activation function or architecture type.")

        self.non_linearity = activations[non_linearity]
        self.architecture = architectures[architecture]
        self.T = T
        self.num_vals = num_vals
        
        # Initialize network layers based on architecture type
        if self.architecture == 1:  # Bottleneck architecture
            ##-- R^{d_input} -> R^{d_hid} layer -- 
            self.fc1_time = nn.Sequential(*[nn.Linear(self.input_dim, self.hidden_dim) for _ in range(self.num_vals)])
            ##-- R^{d_hid} -> R^{d_input} layer --
            self.fc3_time = nn.Sequential(*[nn.Linear(self.hidden_dim, self.input_dim, bias=False) for _ in range(self.num_vals)])
        else:  # Inside or outside architecture
            ##-- R^{d_input} -> R^{d_input} layer --
            self.fc2_time = nn.Sequential(*[nn.Linear(self.input_dim, self.input_dim) for _ in range(self.num_vals)])

    def forward(self, t, x):
        """
        Compute the dynamics f(x(t), t) based on the current time t and state x.
        This represents one step of evaluating the vector field.
        
        Parameters:
        - t (float): Current time
        - x (Tensor): Current state
        
        Returns:
        - Tensor: The derivative dx/dt at the current state and time
        """
        # Determine which time-dependent parameters to use based on current time
        delta_t = self.T / self.num_vals
        k = int(t / delta_t)
        k = min(k, self.num_vals - 1)  # Ensure k is within bounds

        # Apply the appropriate architecture
        if self.architecture == 1:  # Bottleneck architecture
            # Encode -> Nonlinearity -> Decode
            x = self.non_linearity(self.fc1_time[k](x))
            return self.fc3_time[k](x)
        else:  # Inside or outside architecture
            fc2 = self.fc2_time[k]
            # Inside: Apply nonlinearity after linear transformation
            # Outside: Apply nonlinearity before linear transformation
            return self.non_linearity(fc2(x)) if self.architecture == -1 else fc2(self.non_linearity(x))
        
class Semiflow(nn.Module):
    """
    Computes the semiflow x(t) by solving the ODE: x'(t) = f(t, x(t))
    Supports both standard and adjoint ODE integration methods.
    
    The adjoint method is memory-efficient for backpropagation through the ODE solver.
    """
    def __init__(self, device, dynamics, adjoint=False, T=10, step_size=0.1, method='euler', seed_params=1):
        super(Semiflow, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.dynamics = dynamics
        self.T = T
        self.dt = step_size
        self.method = method
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed_params)
        torch.cuda.manual_seed_all(seed_params)
        
    def forward(self, x, eval_times=None):
        """
        Integrates the dynamics from t=0 to t=T (or specified evaluation times).
        
        Parameters:
        - x (Tensor): Initial state
        - eval_times (Tensor, optional): Times at which to evaluate the solution
        
        Returns:
        - Tensor: Solution of the ODE at final time T (or at specified evaluation times)
        """
        # Define integration time points
        if eval_times is None:
            integration_time = torch.tensor([0, self.T]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        # Choose integration method (adjoint or standard)
        if self.adjoint:  
            # Memory-efficient but computationally more expensive
            out = odeint_adjoint(self.dynamics, x, integration_time, 
                                method=self.method, options={'step_size': self.dt})
            # Alternative with adaptive time stepping:
            # out = odeint_adjoint(self.dynamics, x, integration_time, rtol=0.1, atol=0.1, 
            #                      method='dopri5', options={'max_num_steps': MAX_NUM_STEPS})
        else:   
            # Standard ODE integration
            out = odeint(self.dynamics, x, integration_time, 
                        method=self.method, options={'step_size': self.dt})                    
            # Alternative with adaptive time stepping:
            # out = odeint(self.dynamics, x, integration_time, rtol=0.1, atol=0.1, 
            #             method='dopri5', options={'max_num_steps': MAX_NUM_STEPS})

        # Return final state or full trajectory
        if eval_times is None:
            if out is not None:
                return out[1]  # Return final state only
        else:
            return out  # Return full trajectory at evaluation times

    def trajectory(self, x, timesteps):
        """
        Returns the full trajectory of the state over time.
        
        Parameters:
        - x (Tensor): Initial state
        - timesteps (int): Number of time steps for the trajectory
        
        Returns:
        - Tensor: Trajectory of shape (timesteps, batch_size, state_dim)
        """
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.forward(x, eval_times=integration_time)

class NeuralODE(nn.Module):
    """
    Neural ODE implementation that encapsulates the dynamics, semiflow, and optional final layer.
    This is the complete model used for classification.
    """
    def __init__(self, device, fixed_projector, input_dim, hidden_dim, output_dim=2, non_linearity='tanh',
                 adjoint=False, architecture='inside', T=10, num_vals=10, step_size=0.1, method='euler', 
                 final_layer=False, seed_params=1, dynamics_class=None, reg_lambda=0.0):
        super(NeuralODE, self).__init__()
        # Store parameters
        self.device = device
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.T, self.num_vals, self.dt, self.sigma = T, num_vals, step_size, non_linearity
        self.architecture, self.fixed_projector = architecture, fixed_projector
        self.non_linearity = activations[non_linearity]
        self.method, self.adjoint, self.seed_params = method, adjoint, seed_params
        self.final_layer = final_layer
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed_params)
        torch.cuda.manual_seed_all(self.seed_params)

        # Initialize model state
        self.best_param = []
        self.trained = False
        
        # Create dynamics based on custom class or default
        if dynamics_class is None:
            self.fwd_dynamics = Dynamics(device, input_dim, hidden_dim, non_linearity, 
                                    architecture, self.T, self.num_vals, seed_params) 
        else:
            self.fwd_dynamics = dynamics_class(device, input_dim, hidden_dim, non_linearity, 
                                          architecture, self.T, self.num_vals, seed_params, 
                                          reg_lambda=reg_lambda)
        
        self.flow = Semiflow(device, self.fwd_dynamics, adjoint, T, step_size, method, seed_params)
        
        # Optional final layer for classification
        if self.final_layer:
            if not fixed_projector:
                self.linear_layer = nn.Linear(self.flow.dynamics.input_dim, self.output_dim)
            else: 
                self.linear_layer = fixed_projector
            
    def forward(self, x, return_features=False, i=None):
        """
        Forward pass through the Neural ODE.
        
        Parameters:
        - x (Tensor): Input data
        - return_features (bool): Whether to return intermediate features
        - i (int, optional): Index of the best parameters to load
        
        Returns:
        - Tensor: Model predictions
        - Tensor: Trajectory of the ODE flow
        - Tensor (optional): Intermediate features if return_features=True
        """
        # Optionally load saved best parameters
        if self.trained and i is not None:
            self.load_state_dict(self.best_param[i])
            # Recreate dynamics and flow with the loaded parameters
            self.fwd_dynamics = Dynamics(self.device, self.input_dim, self.hidden_dim, 
                                        self.sigma, self.architecture, self.T, 
                                        self.num_vals, self.seed_params) 
            self.flow = Semiflow(self.device, self.fwd_dynamics, self.adjoint, 
                                self.T, self.dt, self.method, self.seed_params)

        # Compute the final state and full trajectory
        features = self.flow(x)
        traj = self.flow.trajectory(x, int(self.T / self.dt) + 1)
        
        # Apply final classification layer if specified
        if self.final_layer:
            pred = self.linear_layer(features)
            if self.output_dim == 1:
                # Binary classification (single output)
                pred = pred.squeeze()
                if return_features:
                    return features, pred, traj
                return pred, traj
            else:
                # Multi-class classification
                proj_traj = self.linear_layer(traj).squeeze()
                if return_features:
                    return features, pred, proj_traj
                return pred, proj_traj
        else:
            # No final layer, return features directly
            pred = features
            proj_traj = traj
            if return_features:
                return features, pred, traj
            return pred, traj
            
            
class adj_Dynamics(nn.Module):
    """
    Defines the adjoint dynamics for neural ODEs: dp(t)/dt = -D_x f(t, x(t))^T p(t)
    
    This implements the vector field for the adjoint ODE, which is used to compute
    gradients with respect to the parameters efficiently.
    """
    def __init__(self, dynamics, x_traj, non_linearity=nn.Tanh(), seed_params=1):
        super(adj_Dynamics, self).__init__()
        self.fwd_dynamics = dynamics
        self.x_traj = x_traj
        
        # Get the derivative of the activation function
        if isinstance(non_linearity, str):
            non_linearity_name = non_linearity
        else:
            non_linearity_name = non_linearity.__class__.__name__.lower()
            
        # Get corresponding derivative function
        derivative_name = f"{non_linearity_name}_prime"
        self.non_linearity = activations.get(derivative_name)

        if self.non_linearity is None:
            raise ValueError(f"Derivative for activation function {non_linearity} not found.")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed_params)
        torch.cuda.manual_seed_all(seed_params)

  
    def forward(self, t, p):
        """
        Compute the adjoint dynamics at time t with adjoint state p.
        
        Parameters:
        - t (float): Current time
        - p (Tensor): Current adjoint state
        
        Returns:
        - Tensor: The derivative dp/dt at the current state and time
        """
        # Determine which time-dependent parameters to use (going backward in time)
        num_vals = self.fwd_dynamics.num_vals
        delta_t = self.fwd_dynamics.T / num_vals
        k = int(t / delta_t)
        k = min(k, num_vals - 1)  # Ensure k is within bounds
        
        # Get the state from the forward trajectory (in reverse time order)
        x = self.x_traj[num_vals - k - 1]
        
        # Compute the gradient application based on architecture
        if self.fwd_dynamics.architecture < 1:
            # Accessing backward-in-time parameters
            w_t = self.fwd_dynamics.fc2_time[num_vals - k - 1].weight
            b_t = self.fwd_dynamics.fc2_time[num_vals - k - 1].bias
            
            if self.fwd_dynamics.architecture == -1:
                # Inside architecture: derivative of σ(Wx + b)
                x = torch.matmul(x, w_t.t()) + b_t
                x_w = self.non_linearity(x)
                grad = torch.matmul(w_t.t(), torch.diag_embed(x_w))
            elif self.fwd_dynamics.architecture == 0:
                # Outside architecture: derivative of W·σ(x)
                x_w = self.non_linearity(x)
                grad = torch.matmul(w_t.t(), torch.diag_embed(x_w))
                
            # Apply the gradient to the co-state vector p
            return -torch.matmul(grad, p.unsqueeze(-1)).squeeze()
        else:
            # Bottleneck architecture
            w_t = self.fwd_dynamics.fc3_time[num_vals - k - 1].weight
            a_t = self.fwd_dynamics.fc1_time[num_vals - k - 1].weight
            b_t = self.fwd_dynamics.fc1_time[num_vals - k - 1].bias
            
            # Compute gradient for bottleneck architecture
            x_w = torch.matmul(x, a_t.t()) + b_t
            x_w = self.non_linearity(x_w)
            x_w = torch.matmul(a_t.t(), torch.diag_embed(x_w))
            grad = torch.matmul(x_w, w_t.t())
            
            return -torch.matmul(grad, p.unsqueeze(-1)).squeeze()


# Enhanced methods for improved functionality

def create_adaptive_ode_solver(dynamics, rtol=1e-3, atol=1e-4, method='dopri5'):
    """
    Creates an ODE solver with adaptive step size for improved accuracy and efficiency.
    
    Parameters:
    - dynamics: The dynamics function
    - rtol: Relative tolerance for adaptive stepping
    - atol: Absolute tolerance for adaptive stepping
    - method: Integration method ('dopri5', 'rk4', etc.)
    
    Returns:
    - solver: A function that solves the ODE
    """
    def solver(x0, t):
        """Solves the ODE from x0 at times t"""
        return odeint(dynamics, x0, t, rtol=rtol, atol=atol, method=method)
    
    return solver

class RegularizedDynamics(Dynamics):
    """
    Dynamics with L2 regularization to prevent overfitting.
    """
    def __init__(self, device, input_dim, hidden_dim, non_linearity='tanh', architecture='inside', 
                 T=10, num_vals=10, seed_params=1, reg_lambda=0.01):
        super(RegularizedDynamics, self).__init__(
            device, input_dim, hidden_dim, non_linearity, architecture, T, num_vals, seed_params
        )
        self.reg_lambda = reg_lambda
        
    def forward(self, t, x):
        """
        Forward pass with L2 regularization.
        """
        # Get the standard dynamics output
        out = super().forward(t, x)
        
        # Add regularization based on the norm of the output
        reg_term = self.reg_lambda * torch.norm(out, p=2, dim=1, keepdim=True)
        return out - reg_term * torch.sign(out)