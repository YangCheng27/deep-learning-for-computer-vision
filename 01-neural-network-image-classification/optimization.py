from typing import Any, Dict, Optional
import numpy as np

def sgd(w, dw, config: Optional[Dict[str, Any]] = None):
    """
    Performs vanilla stochastic gradient descent.

    config is a dict consisting of:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config

def adam(w, dw, config: Optional[Dict[str, Any]] = None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the gradient
    and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid division by zero.
    - m: Moving average of gradient (initially set to zeros, same shape as w).
    - v: Moving average of squared gradient (initially set to zeros, same shape as w).
    - t: Iteration number (initially set to 0).
    """
    if config is None:
        config = {}
    
    # Default values for the Adam optimizer
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))  # First moment vector
    config.setdefault('v', np.zeros_like(w))  # Second moment vector
    config.setdefault('t', 0)  # Timestep

    # Update timestep
    config['t'] += 1
    t = config['t']
    
    # Extract hyperparameters
    lr = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    m = config['m']
    v = config['v']
    
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * dw
    
    # Update biased second moment estimate
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    
    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1 ** t)
    
    # Compute bias-corrected second moment estimate
    v_hat = v / (1 - beta2 ** t)
    
    # Update weights
    w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    # Store updated first and second moments back into the config
    config['m'] = m
    config['v'] = v
    
    return w, config
