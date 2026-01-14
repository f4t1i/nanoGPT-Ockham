"""
Ockham Compressor Controller

A dynamic controller that adjusts hyperparameters based on training metrics,
inspired by audio compressors (Threshold, Ratio, Attack, Release).

This implements the H2 approach: a separate controller that regulates
training dynamics without requiring a full agent (no LLM/RL).
"""

import torch
from typing import Dict, Optional


class OckhamCompressor:
    """
    Dynamic hyperparameter controller inspired by audio compressors.
    
    Audio Compressor Mapping:
    - Threshold: Minimum complexity_cost to trigger compression
    - Ratio: How much to increase lambda_ockham when threshold is exceeded
    - Attack: How quickly to increase compression (lambda_ockham up)
    - Release: How quickly to decrease compression (lambda_ockham down)
    - Make-up Gain: Boost learning rate when model is stable
    
    Args:
        lambda_ockham_base: Base value for lambda_ockham
        threshold: Complexity cost threshold to trigger compression
        ratio: Compression ratio (how much to increase lambda when over threshold)
        attack: Speed of compression increase (0.0 to 1.0)
        release: Speed of compression decrease (0.0 to 1.0)
        lr_makeup_gain: Learning rate boost when model is stable
        min_lambda: Minimum allowed lambda_ockham
        max_lambda: Maximum allowed lambda_ockham
    """
    
    def __init__(
        self,
        lambda_ockham_base: float = 0.01,
        threshold: float = 0.1,
        ratio: float = 2.0,
        attack: float = 0.1,
        release: float = 0.05,
        lr_makeup_gain: float = 1.0,
        min_lambda: float = 0.001,
        max_lambda: float = 0.5,
    ):
        self.lambda_ockham_base = lambda_ockham_base
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.lr_makeup_gain = lr_makeup_gain
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # Internal state
        self.current_lambda = lambda_ockham_base
        self.current_lr_scale = 1.0
        self.compression_active = False
        
        # History for smoothing
        self.complexity_history = []
        self.max_history_len = 10
    
    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update hyperparameters based on current metrics.
        
        Args:
            metrics: Dictionary containing:
                - complexity_cost: Current complexity cost
                - task_loss: Current task loss
                - updated: Whether update was performed
        
        Returns:
            Dictionary with adjusted hyperparameters:
                - lambda_ockham: Adjusted Ockham penalty
                - lr_scale: Learning rate multiplier
                - compression_active: Whether compression is active
        """
        complexity_cost = metrics.get('complexity_cost', 0.0)
        updated = metrics.get('updated', True)
        
        # Track complexity history
        if updated:
            self.complexity_history.append(complexity_cost)
            if len(self.complexity_history) > self.max_history_len:
                self.complexity_history.pop(0)
        
        # Compute smoothed complexity (like RMS in audio)
        if self.complexity_history:
            smoothed_complexity = sum(self.complexity_history) / len(self.complexity_history)
        else:
            smoothed_complexity = complexity_cost
        
        # Determine if compression should be active
        over_threshold = smoothed_complexity > self.threshold
        
        # Attack/Release dynamics
        if over_threshold:
            # Attack: increase compression (increase lambda)
            target_lambda = self.lambda_ockham_base * self.ratio
            self.current_lambda += self.attack * (target_lambda - self.current_lambda)
            self.compression_active = True
            
            # Reduce learning rate when compressing (like reducing gain)
            target_lr_scale = 1.0 / self.ratio
            self.current_lr_scale += self.attack * (target_lr_scale - self.current_lr_scale)
        else:
            # Release: decrease compression (decrease lambda)
            target_lambda = self.lambda_ockham_base
            self.current_lambda += self.release * (target_lambda - self.current_lambda)
            self.compression_active = False
            
            # Make-up gain: boost learning rate when stable
            target_lr_scale = self.lr_makeup_gain
            self.current_lr_scale += self.release * (target_lr_scale - self.current_lr_scale)
        
        # Clamp lambda to valid range
        self.current_lambda = max(self.min_lambda, min(self.max_lambda, self.current_lambda))
        
        # Clamp lr_scale to reasonable range
        self.current_lr_scale = max(0.1, min(2.0, self.current_lr_scale))
        
        return {
            'lambda_ockham': self.current_lambda,
            'lr_scale': self.current_lr_scale,
            'compression_active': self.compression_active,
            'smoothed_complexity': smoothed_complexity,
        }
    
    def reset(self):
        """Reset controller to initial state."""
        self.current_lambda = self.lambda_ockham_base
        self.current_lr_scale = 1.0
        self.compression_active = False
        self.complexity_history = []
    
    def get_state(self) -> Dict:
        """Get current controller state for logging."""
        return {
            'current_lambda': self.current_lambda,
            'current_lr_scale': self.current_lr_scale,
            'compression_active': self.compression_active,
            'complexity_history_len': len(self.complexity_history),
        }
    
    def __repr__(self):
        return (
            f"OckhamCompressor("
            f"threshold={self.threshold:.3f}, "
            f"ratio={self.ratio:.1f}, "
            f"attack={self.attack:.2f}, "
            f"release={self.release:.2f}, "
            f"current_lambda={self.current_lambda:.4f}, "
            f"compression_active={self.compression_active})"
        )


class AdaptiveOckhamCompressor(OckhamCompressor):
    """
    Extended compressor that also adapts surprise_threshold dynamically.
    
    This is useful for handling varying data distributions where the
    "loudness" (task loss) changes over time.
    """
    
    def __init__(
        self,
        surprise_threshold_base: float = 2.0,
        surprise_adaptation_rate: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.surprise_threshold_base = surprise_threshold_base
        self.surprise_adaptation_rate = surprise_adaptation_rate
        self.current_surprise_threshold = surprise_threshold_base
        
        # Track loss history
        self.loss_history = []
    
    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Update both lambda and surprise_threshold."""
        # Get base updates
        result = super().update(metrics)
        
        # Track task loss
        task_loss = metrics.get('task_loss', 0.0)
        if metrics.get('updated', True):
            self.loss_history.append(task_loss)
            if len(self.loss_history) > self.max_history_len:
                self.loss_history.pop(0)
        
        # Adapt surprise threshold based on loss distribution
        if self.loss_history:
            # Use median as robust estimate of "typical" loss
            sorted_losses = sorted(self.loss_history)
            median_loss = sorted_losses[len(sorted_losses) // 2]
            
            # Adapt threshold to be slightly above median
            target_threshold = median_loss * 1.2
            self.current_surprise_threshold += self.surprise_adaptation_rate * (
                target_threshold - self.current_surprise_threshold
            )
            
            # Clamp to reasonable range
            self.current_surprise_threshold = max(
                0.1, min(10.0, self.current_surprise_threshold)
            )
        
        result['surprise_threshold'] = self.current_surprise_threshold
        return result
    
    def reset(self):
        """Reset controller to initial state."""
        super().reset()
        self.current_surprise_threshold = self.surprise_threshold_base
        self.loss_history = []


if __name__ == "__main__":
    # Demo: Show compressor behavior
    print("=" * 80)
    print("OCKHAM COMPRESSOR DEMONSTRATION")
    print("=" * 80)
    
    compressor = OckhamCompressor(
        lambda_ockham_base=0.01,
        threshold=0.1,
        ratio=2.0,
        attack=0.2,
        release=0.05,
    )
    
    print(f"\nInitial state: {compressor}")
    print("\nSimulating training with varying complexity costs...")
    print("-" * 80)
    
    # Simulate varying complexity costs
    test_complexities = [
        0.05, 0.08, 0.12, 0.15, 0.18,  # Rising (attack)
        0.16, 0.14, 0.11, 0.08, 0.05,  # Falling (release)
        0.03, 0.02, 0.02, 0.03, 0.04,  # Stable low
    ]
    
    for i, complexity in enumerate(test_complexities, 1):
        metrics = {
            'complexity_cost': complexity,
            'task_loss': 1.5,
            'updated': True,
        }
        
        result = compressor.update(metrics)
        
        status = "🔴 COMPRESSING" if result['compression_active'] else "🟢 RELEASED"
        print(
            f"Step {i:2d}: complexity={complexity:.3f} → "
            f"λ={result['lambda_ockham']:.4f}, "
            f"lr_scale={result['lr_scale']:.2f} {status}"
        )
    
    print("-" * 80)
    print(f"\nFinal state: {compressor}")
    print("\n✓ Demo complete!")
