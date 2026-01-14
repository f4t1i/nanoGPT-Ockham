"""
Core Learning Plugins V2 for nanoGPT-Ockham

Refactored to use OccamContext instead of TrainingState.
This separates mechanics (code) from policy (config).
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from ockham_context import OccamContext


class OccamPlugin(ABC):
    """
    Base class for all Ockham plugins (V2).
    
    Uses OccamContext instead of TrainingState for cleaner separation
    of mechanics and policy.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def on_batch_start(self, ctx: OccamContext) -> OccamContext:
        """
        Called before processing a batch.
        
        Plugins can modify hyperparameters here based on current metrics.
        
        Args:
            ctx: Current Ockham context
        
        Returns:
            Modified Ockham context
        """
        pass
    
    def on_batch_end(self, ctx: OccamContext) -> None:
        """
        Called after processing a batch.
        
        Plugins can update internal state or log metrics here.
        
        Args:
            ctx: Current Ockham context with updated metrics
        """
        pass
    
    def on_consolidate(self, ctx: OccamContext) -> None:
        """
        Called when model consolidates (anchor is updated).
        
        Args:
            ctx: Current Ockham context
        """
        pass
    
    def reset(self) -> None:
        """Reset plugin to initial state."""
        pass
    
    def get_state(self) -> Dict[str, any]:
        """Get plugin state for logging/debugging."""
        return {
            'name': self.name,
            'enabled': self.enabled,
        }
    
    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"


class OckhamGatePlugin(OccamPlugin):
    """
    Implements the core Ockham surprise gate.
    
    Skips updates when task_loss < surprise_threshold.
    """
    
    def __init__(
        self,
        name: str = "ockham_gate",
        surprise_threshold: float = 2.0,
        adaptive: bool = False,
        adaptation_rate: float = 0.01,
    ):
        super().__init__(name)
        self.surprise_threshold = surprise_threshold
        self.adaptive = adaptive
        self.adaptation_rate = adaptation_rate
        
        self.loss_history = []
        self.max_history_len = 100
        self.skip_count = 0
        self.update_count = 0
    
    def on_batch_start(self, ctx: OccamContext) -> OccamContext:
        # Adapt threshold if enabled
        if self.adaptive and len(self.loss_history) >= 10:
            sorted_losses = sorted(self.loss_history)
            median = sorted_losses[len(sorted_losses) // 2]
            target_threshold = median * 1.2
            
            self.surprise_threshold += self.adaptation_rate * (
                target_threshold - self.surprise_threshold
            )
            self.surprise_threshold = max(0.1, min(10.0, self.surprise_threshold))
        
        # Update context threshold
        ctx.surprise_threshold = self.surprise_threshold
        
        # Check if update should be skipped
        if ctx.task_loss < self.surprise_threshold:
            ctx.updated = False
            self.skip_count += 1
        else:
            ctx.updated = True
            self.update_count += 1
        
        # Store for logging
        ctx.custom[f'{self.name}_skip_count'] = self.skip_count
        ctx.custom[f'{self.name}_update_count'] = self.update_count
        if self.update_count + self.skip_count > 0:
            update_rate = self.update_count / (self.update_count + self.skip_count)
            ctx.custom[f'{self.name}_update_rate'] = update_rate
            ctx.update_rate = update_rate
        
        return ctx
    
    def on_batch_end(self, ctx: OccamContext) -> None:
        # Track loss history
        if ctx.updated:
            self.loss_history.append(ctx.task_loss)
            if len(self.loss_history) > self.max_history_len:
                self.loss_history.pop(0)
    
    def reset(self):
        self.loss_history = []
        self.skip_count = 0
        self.update_count = 0
    
    def get_state(self) -> Dict:
        state = super().get_state()
        update_rate = 0.0
        if self.update_count + self.skip_count > 0:
            update_rate = self.update_count / (self.update_count + self.skip_count)
        
        state.update({
            'surprise_threshold': self.surprise_threshold,
            'adaptive': self.adaptive,
            'skip_count': self.skip_count,
            'update_count': self.update_count,
            'update_rate': update_rate,
        })
        return state


class CompressorPlugin(OccamPlugin):
    """
    Audio compressor-inspired dynamic hyperparameter control.
    
    Automatically adjusts lambda_ockham and learning_rate based on
    complexity_cost.
    """
    
    def __init__(
        self,
        name: str = "compressor",
        threshold: float = 0.1,
        ratio: float = 2.0,
        attack: float = 0.1,
        release: float = 0.05,
        makeup_gain: float = 1.0,
        min_lambda: float = 0.001,
        max_lambda: float = 0.5,
    ):
        super().__init__(name)
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.makeup_gain = makeup_gain
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # Internal state
        self.base_lambda = None
        self.base_lr = None
        self.current_lambda_scale = 1.0
        self.current_lr_scale = 1.0
        self.complexity_history = []
        self.max_history_len = 10
    
    def on_batch_start(self, ctx: OccamContext) -> OccamContext:
        # Initialize base values on first call
        if self.base_lambda is None:
            self.base_lambda = ctx.lambda_ockham
            self.base_lr = ctx.learning_rate
        
        # Track complexity history
        if ctx.updated:
            self.complexity_history.append(ctx.complexity_cost)
            if len(self.complexity_history) > self.max_history_len:
                self.complexity_history.pop(0)
        
        # Compute smoothed complexity (RMS-like)
        if self.complexity_history:
            smoothed = sum(self.complexity_history) / len(self.complexity_history)
        else:
            smoothed = ctx.complexity_cost
        
        # Determine compression
        over_threshold = smoothed > self.threshold
        
        if over_threshold:
            # Attack: increase compression
            target_lambda_scale = self.ratio
            target_lr_scale = 1.0 / self.ratio
            
            self.current_lambda_scale += self.attack * (target_lambda_scale - self.current_lambda_scale)
            self.current_lr_scale += self.attack * (target_lr_scale - self.current_lr_scale)
        else:
            # Release: decrease compression, apply makeup gain
            target_lambda_scale = 1.0
            target_lr_scale = self.makeup_gain
            
            self.current_lambda_scale += self.release * (target_lambda_scale - self.current_lambda_scale)
            self.current_lr_scale += self.release * (target_lr_scale - self.current_lr_scale)
        
        # Apply scaling
        new_lambda = self.base_lambda * self.current_lambda_scale
        new_lr = self.base_lr * self.current_lr_scale
        
        # Clamp to valid ranges
        ctx.lambda_ockham = max(self.min_lambda, min(self.max_lambda, new_lambda))
        ctx.learning_rate = max(1e-6, min(1e-2, new_lr))
        
        # Store smoothed complexity for logging
        ctx.custom[f'{self.name}_smoothed_complexity'] = smoothed
        ctx.custom[f'{self.name}_compressing'] = over_threshold
        
        return ctx
    
    def reset(self):
        self.base_lambda = None
        self.base_lr = None
        self.current_lambda_scale = 1.0
        self.current_lr_scale = 1.0
        self.complexity_history = []
    
    def get_state(self) -> Dict:
        state = super().get_state()
        state.update({
            'threshold': self.threshold,
            'ratio': self.ratio,
            'current_lambda_scale': self.current_lambda_scale,
            'current_lr_scale': self.current_lr_scale,
        })
        return state


class LimiterPlugin(OccamPlugin):
    """
    Hard limiter for complexity cost and gradient norm.
    
    Prevents model from drifting too far from anchor or having exploding gradients.
    """
    
    def __init__(
        self,
        name: str = "limiter",
        complexity_ceiling: float = 0.2,
        grad_norm_ceiling: float = 1.0,
        force_consolidate: bool = True,
    ):
        super().__init__(name)
        self.complexity_ceiling = complexity_ceiling
        self.grad_norm_ceiling = grad_norm_ceiling
        self.force_consolidate = force_consolidate
        
        self.complexity_hits = 0
        self.grad_norm_hits = 0
    
    def on_batch_start(self, ctx: OccamContext) -> OccamContext:
        # Limiter doesn't modify hyperparameters on batch start
        return ctx
    
    def on_batch_end(self, ctx: OccamContext) -> None:
        # Check complexity ceiling
        if ctx.complexity_cost > self.complexity_ceiling:
            self.complexity_hits += 1
            if self.force_consolidate:
                ctx.consolidating = True
                ctx.custom[f'{self.name}_consolidate_triggered'] = True
        
        # Check gradient norm ceiling (informational only)
        if ctx.grad_norm > self.grad_norm_ceiling:
            self.grad_norm_hits += 1
        
        # Store hit counts
        ctx.custom[f'{self.name}_complexity_hits'] = self.complexity_hits
        ctx.custom[f'{self.name}_grad_norm_hits'] = self.grad_norm_hits
    
    def reset(self):
        self.complexity_hits = 0
        self.grad_norm_hits = 0
    
    def get_state(self) -> Dict:
        state = super().get_state()
        state.update({
            'complexity_ceiling': self.complexity_ceiling,
            'grad_norm_ceiling': self.grad_norm_ceiling,
            'complexity_hits': self.complexity_hits,
            'grad_norm_hits': self.grad_norm_hits,
        })
        return state


if __name__ == "__main__":
    print("=" * 80)
    print("PLUGINS V2 DEMONSTRATION (with OccamContext)")
    print("=" * 80)
    
    # Create plugins
    gate = OckhamGatePlugin(surprise_threshold=1.5)
    compressor = CompressorPlugin(threshold=0.1, ratio=2.0)
    limiter = LimiterPlugin(complexity_ceiling=0.2)
    
    plugins = [gate, compressor, limiter]
    
    print(f"\nPlugin chain: {[p.name for p in plugins]}")
    print("\nSimulating training with varying conditions...")
    print("-" * 80)
    
    # Simulate training
    for i in range(20):
        ctx = OccamContext(
            iter_num=i,
            task_loss=2.0 + 0.5 * np.sin(i * 0.3),  # Oscillating loss
            learning_rate=1e-3,
            lambda_ockham=0.01,
            surprise_threshold=2.0,
            complexity_cost=0.05 + i * 0.01,  # Increasing complexity
            grad_norm=0.5 + 0.2 * np.random.randn(),
        )
        
        # Process batch start (plugins modify hyperparameters)
        for plugin in plugins:
            if plugin.enabled:
                ctx = plugin.on_batch_start(ctx)
        
        # Process batch end (plugins react to metrics)
        for plugin in plugins:
            if plugin.enabled:
                plugin.on_batch_end(ctx)
        
        # Log every 5 iterations
        if i % 5 == 0:
            updated_str = "✓" if ctx.updated else "✗"
            compressing = ctx.custom.get('compressor_compressing', False)
            comp_str = "🔴" if compressing else "🟢"
            
            print(
                f"Iter {i:2d}: loss={ctx.task_loss:.2f} {updated_str}, "
                f"complexity={ctx.complexity_cost:.3f} {comp_str}, "
                f"λ={ctx.lambda_ockham:.4f}, "
                f"lr={ctx.learning_rate:.6f}"
            )
    
    print("-" * 80)
    print("\n✓ Demo complete!")
    print("\nFinal plugin states:")
    for plugin in plugins:
        print(f"  {plugin.name}: {plugin.get_state()}")
