"""
OckhamLearner: Intelligent Test-Time Training with Ockham's Razor

This module implements the core OckhamLearner class, which wraps a PyTorch model
and enables controlled, minimal adaptation during test-time training (TTT).

Key Principles:
1. Minimal Parameter Change: Only update weights when absolutely necessary
2. Surprise-Driven Learning: Use a "surprise threshold" to gate updates
3. Anchor Stability: Regularize against drift from a stable anchor point

The OckhamLearner operationalizes Ockham's Razor: "Entities should not be 
multiplied beyond necessity" - applied to parameter updates.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import copy


class OckhamLearner:
    """
    A wrapper around a PyTorch model that implements Ockham's Razor principles
    for test-time training and adaptation.
    
    The learner maintains an "anchor" state (θ_anchor) and penalizes deviations
    from it, ensuring that the model only adapts when the task loss (surprise)
    justifies the complexity cost.
    
    Loss Function:
        L_total = L_task + λ_ockham * Ω(Δθ)
        
    where:
        - L_task: Task-specific loss (e.g., cross-entropy)
        - λ_ockham: Regularization strength (how strongly to resist change)
        - Ω(Δθ): Complexity penalty (squared L2 norm of parameter changes)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_ockham: float = 0.01,
        surprise_threshold: Optional[float] = None,
        device: str = 'cuda'
    ):
        """
        Initialize the OckhamLearner.
        
        Args:
            model: The PyTorch model to wrap
            optimizer: The optimizer to use for updates
            lambda_ockham: Regularization strength for anchor penalty (default: 0.01)
            surprise_threshold: Minimum task loss to trigger an update (default: None, always update)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.lambda_ockham = lambda_ockham
        self.surprise_threshold = surprise_threshold
        self.device = device
        
        # Store the initial anchor state
        self.theta_anchor = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        # Metrics tracking
        self.metrics = {
            'total_batches': 0,
            'updates_performed': 0,
            'updates_skipped': 0,
            'total_task_loss': 0.0,
            'total_complexity_cost': 0.0,
        }
    
    def _calculate_ockham_loss(self) -> torch.Tensor:
        """
        Calculate the Ockham complexity penalty: Ω(Δθ) = Σ ||θ - θ_anchor||²
        
        This measures how far the current parameters have drifted from the anchor.
        Uses squared L2 norm for numerical stability and smooth gradients.
        
        Returns:
            Scalar tensor representing the complexity cost
        """
        complexity_cost = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.theta_anchor:
                diff = param - self.theta_anchor[name].to(self.device)
                complexity_cost += torch.sum(diff * diff)
        
        return complexity_cost
    
    def adapt(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: callable
    ) -> Dict[str, float]:
        """
        Perform one adaptation step with Ockham's Razor principles.
        
        This is the core method that implements the "Surprise Gate":
        1. Compute task loss (surprise)
        2. If surprise < threshold, skip update (save computation)
        3. Otherwise, compute total loss with Ockham penalty and update
        
        Args:
            inputs: Input tensor for the model
            targets: Target tensor for loss computation
            loss_fn: Loss function that takes (logits, targets) and returns scalar loss
            
        Returns:
            Dictionary with metrics:
                - 'task_loss': The task-specific loss value
                - 'complexity_cost': The Ockham penalty value
                - 'total_loss': Combined loss (if update was performed)
                - 'updated': Boolean indicating if weights were updated
        """
        self.metrics['total_batches'] += 1
        
        # Forward pass to compute task loss (surprise)
        self.model.train()
        logits = self.model(inputs)
        task_loss = loss_fn(logits, targets)
        
        task_loss_value = task_loss.item()
        self.metrics['total_task_loss'] += task_loss_value
        
        # Surprise Gate: Check if update is warranted
        if self.surprise_threshold is not None and task_loss_value < self.surprise_threshold:
            self.metrics['updates_skipped'] += 1
            return {
                'task_loss': task_loss_value,
                'complexity_cost': 0.0,
                'total_loss': task_loss_value,
                'updated': False
            }
        
        # Update is warranted - compute Ockham penalty
        complexity_cost = self._calculate_ockham_loss()
        complexity_cost_value = complexity_cost.item()
        self.metrics['total_complexity_cost'] += complexity_cost_value
        
        # Total loss with Ockham regularization
        total_loss = task_loss + self.lambda_ockham * complexity_cost
        
        # Backward pass and optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.metrics['updates_performed'] += 1
        
        return {
            'task_loss': task_loss_value,
            'complexity_cost': complexity_cost_value,
            'total_loss': total_loss.item(),
            'updated': True
        }
    
    def consolidate(self):
        """
        Consolidate the current model state as the new anchor point.
        
        This should be called when the model has reached a new stable state
        that we want to preserve. After consolidation, the complexity cost
        resets to zero, and future adaptations will be measured relative to
        this new anchor.
        
        Use cases:
        - After successful adaptation to a new domain
        - When validation loss reaches a new minimum
        - At regular intervals during long-running adaptation
        """
        self.theta_anchor = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        print(f"[OckhamLearner] Anchor consolidated at batch {self.metrics['total_batches']}")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics and statistics.
        
        Returns:
            Dictionary with:
                - total_batches: Total number of batches processed
                - updates_performed: Number of actual weight updates
                - updates_skipped: Number of updates skipped by surprise gate
                - update_rate: Fraction of batches that triggered updates
                - avg_task_loss: Average task loss across all batches
                - avg_complexity_cost: Average complexity cost (when computed)
        """
        update_rate = (
            self.metrics['updates_performed'] / self.metrics['total_batches']
            if self.metrics['total_batches'] > 0 else 0.0
        )
        
        avg_task_loss = (
            self.metrics['total_task_loss'] / self.metrics['total_batches']
            if self.metrics['total_batches'] > 0 else 0.0
        )
        
        avg_complexity_cost = (
            self.metrics['total_complexity_cost'] / self.metrics['updates_performed']
            if self.metrics['updates_performed'] > 0 else 0.0
        )
        
        return {
            'total_batches': self.metrics['total_batches'],
            'updates_performed': self.metrics['updates_performed'],
            'updates_skipped': self.metrics['updates_skipped'],
            'update_rate': update_rate,
            'avg_task_loss': avg_task_loss,
            'avg_complexity_cost': avg_complexity_cost,
        }
    
    def reset_metrics(self):
        """Reset all metrics to zero."""
        self.metrics = {
            'total_batches': 0,
            'updates_performed': 0,
            'updates_skipped': 0,
            'total_task_loss': 0.0,
            'total_complexity_cost': 0.0,
        }
    
    def save_state(self, path: str):
        """
        Save the complete OckhamLearner state.
        
        Args:
            path: Path to save the state dict
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'theta_anchor': self.theta_anchor,
            'metrics': self.metrics,
            'lambda_ockham': self.lambda_ockham,
            'surprise_threshold': self.surprise_threshold,
        }
        torch.save(state, path)
        print(f"[OckhamLearner] State saved to {path}")
    
    def load_state(self, path: str):
        """
        Load a previously saved OckhamLearner state.
        
        Args:
            path: Path to the saved state dict
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.theta_anchor = state['theta_anchor']
        self.metrics = state['metrics']
        self.lambda_ockham = state['lambda_ockham']
        self.surprise_threshold = state['surprise_threshold']
        print(f"[OckhamLearner] State loaded from {path}")
