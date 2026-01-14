"""
Core Learning Plugins for nanoGPT-Ockham

Collection of VST-inspired plugins for training control:
- CompressorPlugin: Dynamic compression of gradient updates
- EQPlugin: Curriculum learning via data weighting
- LimiterPlugin: Hard caps on complexity and gradients
- SaturationPlugin: Controlled noise injection for exploration
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from plugin_system import LearningPlugin, TrainingState


class CompressorPlugin(LearningPlugin):
    """
    Audio compressor-inspired dynamic hyperparameter control.
    
    Automatically adjusts lambda_ockham and learning_rate based on
    complexity_cost, similar to how an audio compressor reduces dynamic range.
    
    Args:
        threshold: Complexity cost threshold to trigger compression
        ratio: Compression ratio (how much to increase lambda when over threshold)
        attack: Speed of compression increase (0.0 to 1.0)
        release: Speed of compression decrease (0.0 to 1.0)
        makeup_gain: Learning rate boost when model is stable
        min_lambda: Minimum allowed lambda_ockham
        max_lambda: Maximum allowed lambda_ockham
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
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        # Initialize base values on first call
        if self.base_lambda is None:
            self.base_lambda = state.lambda_ockham
            self.base_lr = state.learning_rate
        
        # Track complexity history
        if state.updated:
            self.complexity_history.append(state.complexity_cost)
            if len(self.complexity_history) > self.max_history_len:
                self.complexity_history.pop(0)
        
        # Compute smoothed complexity (RMS-like)
        if self.complexity_history:
            smoothed = sum(self.complexity_history) / len(self.complexity_history)
        else:
            smoothed = state.complexity_cost
        
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
        state.lambda_ockham = max(self.min_lambda, min(self.max_lambda, new_lambda))
        state.learning_rate = max(1e-6, min(1e-2, new_lr))
        
        # Store smoothed complexity for logging
        state.custom[f'{self.name}_smoothed_complexity'] = smoothed
        state.custom[f'{self.name}_compressing'] = over_threshold
        
        return state
    
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


class EQPlugin(LearningPlugin):
    """
    Equalizer-inspired curriculum learning plugin.
    
    Adjusts learning focus on different "bands" of data difficulty.
    In practice, this could weight samples differently based on loss.
    
    Args:
        bands: Dictionary mapping difficulty levels to gain values
                e.g., {"easy": 0.5, "medium": 1.0, "hard": 1.5}
        adaptation_rate: How quickly to adjust band gains
    """
    
    def __init__(
        self,
        name: str = "eq",
        bands: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
    ):
        super().__init__(name)
        self.bands = bands or {"easy": 0.5, "medium": 1.0, "hard": 1.5}
        self.adaptation_rate = adaptation_rate
        
        # Track loss distribution to classify difficulty
        self.loss_history = []
        self.max_history_len = 100
        self.current_band = "medium"
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        # Track loss history
        if state.updated:
            self.loss_history.append(state.task_loss)
            if len(self.loss_history) > self.max_history_len:
                self.loss_history.pop(0)
        
        # Classify current batch difficulty
        if len(self.loss_history) >= 10:
            sorted_losses = sorted(self.loss_history)
            percentile_33 = sorted_losses[len(sorted_losses) // 3]
            percentile_66 = sorted_losses[2 * len(sorted_losses) // 3]
            
            if state.task_loss < percentile_33:
                self.current_band = "easy"
            elif state.task_loss < percentile_66:
                self.current_band = "medium"
            else:
                self.current_band = "hard"
        
        # Apply band gain to learning rate
        gain = self.bands.get(self.current_band, 1.0)
        state.learning_rate *= gain
        
        # Store for logging
        state.custom[f'{self.name}_band'] = self.current_band
        state.custom[f'{self.name}_gain'] = gain
        
        return state
    
    def reset(self):
        self.loss_history = []
        self.current_band = "medium"
    
    def get_state(self) -> Dict:
        state = super().get_state()
        state.update({
            'bands': self.bands,
            'current_band': self.current_band,
            'loss_history_len': len(self.loss_history),
        })
        return state


class LimiterPlugin(LearningPlugin):
    """
    Hard limiter for complexity cost and gradient norm.
    
    Prevents model from drifting too far from anchor or having exploding gradients.
    
    Args:
        complexity_ceiling: Maximum allowed complexity_cost
        grad_norm_ceiling: Maximum allowed gradient norm
        force_consolidate: Whether to force consolidation when ceiling is hit
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
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        # Limiter doesn't modify hyperparameters on batch start
        return state
    
    def on_batch_end(self, state: TrainingState) -> None:
        # Check complexity ceiling
        if state.complexity_cost > self.complexity_ceiling:
            self.complexity_hits += 1
            if self.force_consolidate:
                state.consolidating = True
                state.custom[f'{self.name}_consolidate_triggered'] = True
        
        # Check gradient norm ceiling (informational only, clipping happens elsewhere)
        if state.grad_norm > self.grad_norm_ceiling:
            self.grad_norm_hits += 1
        
        # Store hit counts
        state.custom[f'{self.name}_complexity_hits'] = self.complexity_hits
        state.custom[f'{self.name}_grad_norm_hits'] = self.grad_norm_hits
    
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


class SaturationPlugin(LearningPlugin):
    """
    Controlled noise injection for exploration and robustness.
    
    Adds small amounts of noise to gradients or learning rate,
    similar to audio saturation adding harmonics.
    
    Args:
        drive: Amount of noise to inject (0.0 to 1.0)
        noise_type: Where to inject noise ('gradient', 'learning_rate', 'both')
        warmup_iters: Number of iterations before noise is fully active
    """
    
    def __init__(
        self,
        name: str = "saturation",
        drive: float = 0.01,
        noise_type: str = 'learning_rate',
        warmup_iters: int = 100,
    ):
        super().__init__(name)
        self.drive = drive
        self.noise_type = noise_type
        self.warmup_iters = warmup_iters
        
        self.current_drive = 0.0
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        # Warmup: gradually increase drive
        if state.iter_num < self.warmup_iters:
            self.current_drive = self.drive * (state.iter_num / self.warmup_iters)
        else:
            self.current_drive = self.drive
        
        # Apply noise to learning rate
        if self.noise_type in ['learning_rate', 'both']:
            noise = np.random.randn() * self.current_drive
            state.learning_rate *= (1.0 + noise)
            state.learning_rate = max(1e-6, state.learning_rate)  # Keep positive
        
        # Note: Gradient noise would be applied in the training loop itself,
        # not here. We just set a flag for the training loop to use.
        if self.noise_type in ['gradient', 'both']:
            state.custom[f'{self.name}_gradient_noise'] = self.current_drive
        
        state.custom[f'{self.name}_current_drive'] = self.current_drive
        
        return state
    
    def reset(self):
        self.current_drive = 0.0
    
    def get_state(self) -> Dict:
        state = super().get_state()
        state.update({
            'drive': self.drive,
            'noise_type': self.noise_type,
            'current_drive': self.current_drive,
        })
        return state


class OckhamGatePlugin(LearningPlugin):
    """
    Implements the core Ockham surprise gate.
    
    This is a plugin wrapper around the existing OckhamLearner logic.
    Skips updates when task_loss < surprise_threshold.
    
    Args:
        surprise_threshold: Minimum loss to trigger update
        adaptive: Whether to adapt threshold based on loss distribution
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
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        # Adapt threshold if enabled
        if self.adaptive and len(self.loss_history) >= 10:
            sorted_losses = sorted(self.loss_history)
            median = sorted_losses[len(sorted_losses) // 2]
            target_threshold = median * 1.2
            
            self.surprise_threshold += self.adaptation_rate * (
                target_threshold - self.surprise_threshold
            )
            self.surprise_threshold = max(0.1, min(10.0, self.surprise_threshold))
        
        # Update state threshold
        state.surprise_threshold = self.surprise_threshold
        
        # Check if update should be skipped
        if state.task_loss < self.surprise_threshold:
            state.updated = False
            self.skip_count += 1
        else:
            state.updated = True
            self.update_count += 1
        
        # Store for logging
        state.custom[f'{self.name}_skip_count'] = self.skip_count
        state.custom[f'{self.name}_update_count'] = self.update_count
        if self.update_count + self.skip_count > 0:
            update_rate = self.update_count / (self.update_count + self.skip_count)
            state.custom[f'{self.name}_update_rate'] = update_rate
        
        return state
    
    def on_batch_end(self, state: TrainingState) -> None:
        # Track loss history
        if state.updated:
            self.loss_history.append(state.task_loss)
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


if __name__ == "__main__":
    from plugin_system import PluginHost
    
    print("=" * 80)
    print("CORE PLUGINS DEMONSTRATION")
    print("=" * 80)
    
    # Create plugins
    gate = OckhamGatePlugin(surprise_threshold=1.5)
    compressor = CompressorPlugin(threshold=0.1, ratio=2.0)
    limiter = LimiterPlugin(complexity_ceiling=0.2)
    eq = EQPlugin(bands={"easy": 0.5, "medium": 1.0, "hard": 1.5})
    saturation = SaturationPlugin(drive=0.02, warmup_iters=5)
    
    # Create host with plugin chain
    host = PluginHost(plugins=[gate, compressor, eq, limiter, saturation])
    
    print(f"\nPlugin chain: {[p.name for p in host.plugins]}")
    print("\nSimulating training with varying conditions...")
    print("-" * 80)
    
    # Simulate training
    state = TrainingState(
        learning_rate=1e-3,
        lambda_ockham=0.01,
        surprise_threshold=2.0,
    )
    
    # Vary loss and complexity
    for i in range(20):
        state.iter_num = i
        state.task_loss = 2.0 + 0.5 * np.sin(i * 0.3)  # Oscillating loss
        state.complexity_cost = 0.05 + i * 0.01  # Increasing complexity
        state.grad_norm = 0.5 + 0.2 * np.random.randn()
        
        # Process batch start
        state = host.process_batch_start(state)
        
        # Process batch end
        host.process_batch_end(state)
        
        # Log every 5 iterations
        if i % 5 == 0:
            updated_str = "✓" if state.updated else "✗"
            compressing = state.custom.get('compressor_compressing', False)
            comp_str = "🔴" if compressing else "🟢"
            band = state.custom.get('eq_band', 'medium')
            
            print(
                f"Iter {i:2d}: loss={state.task_loss:.2f} {updated_str}, "
                f"complexity={state.complexity_cost:.3f} {comp_str}, "
                f"λ={state.lambda_ockham:.4f}, "
                f"lr={state.learning_rate:.6f}, "
                f"band={band}"
            )
    
    print("-" * 80)
    print("\n✓ Demo complete!")
    print("\nFinal plugin states:")
    for plugin in host.plugins:
        print(f"  {plugin.name}: {plugin.get_state()}")
