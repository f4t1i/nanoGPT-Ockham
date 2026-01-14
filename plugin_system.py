"""
Plugin System for nanoGPT-Ockham

A modular, VST-inspired plugin architecture for training methods.
Allows composable, swappable training behaviors through plugins.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import torch


@dataclass
class TrainingState:
    """
    Encapsulates the current state of training.
    
    This is passed to plugins so they can read metrics and adjust hyperparameters.
    """
    # Current iteration
    iter_num: int = 0
    
    # Hyperparameters (mutable by plugins)
    learning_rate: float = 1e-3
    lambda_ockham: float = 0.01
    surprise_threshold: float = 2.0
    consolidate_interval: int = 1000
    
    # Metrics (read by plugins)
    task_loss: float = 0.0
    complexity_cost: float = 0.0
    grad_norm: float = 0.0
    update_rate: float = 1.0
    
    # Flags
    updated: bool = True
    consolidating: bool = False
    
    # Custom data (for plugin-specific state)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'iter_num': self.iter_num,
            'learning_rate': self.learning_rate,
            'lambda_ockham': self.lambda_ockham,
            'surprise_threshold': self.surprise_threshold,
            'task_loss': self.task_loss,
            'complexity_cost': self.complexity_cost,
            'grad_norm': self.grad_norm,
            'update_rate': self.update_rate,
            'updated': self.updated,
        }


class LearningPlugin(ABC):
    """
    Base class for all learning plugins.
    
    Plugins can modify hyperparameters, react to metrics, and maintain internal state.
    They are chained together in a PluginHost.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        """
        Called before processing a batch.
        
        Plugins can modify hyperparameters here based on current metrics.
        
        Args:
            state: Current training state
        
        Returns:
            Modified training state
        """
        pass
    
    def on_batch_end(self, state: TrainingState) -> None:
        """
        Called after processing a batch.
        
        Plugins can update internal state or log metrics here.
        
        Args:
            state: Current training state with updated metrics
        """
        pass
    
    def on_consolidate(self, state: TrainingState) -> None:
        """
        Called when model consolidates (anchor is updated).
        
        Args:
            state: Current training state
        """
        pass
    
    def reset(self) -> None:
        """Reset plugin to initial state."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get plugin state for logging/debugging."""
        return {
            'name': self.name,
            'enabled': self.enabled,
        }
    
    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"


class PluginHost:
    """
    Manages and chains learning plugins.
    
    Plugins are executed in order:
    1. on_batch_start: Each plugin can modify hyperparameters
    2. Training step happens (outside PluginHost)
    3. on_batch_end: Each plugin can react to metrics
    """
    
    def __init__(self, plugins: Optional[List[LearningPlugin]] = None):
        self.plugins = plugins or []
    
    def add_plugin(self, plugin: LearningPlugin) -> None:
        """Add a plugin to the chain."""
        self.plugins.append(plugin)
    
    def remove_plugin(self, name: str) -> bool:
        """Remove a plugin by name. Returns True if found and removed."""
        for i, plugin in enumerate(self.plugins):
            if plugin.name == name:
                self.plugins.pop(i)
                return True
        return False
    
    def get_plugin(self, name: str) -> Optional[LearningPlugin]:
        """Get a plugin by name."""
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None
    
    def process_batch_start(self, state: TrainingState) -> TrainingState:
        """
        Process all plugins before batch.
        
        Plugins are executed in order, each receiving the state modified by previous plugins.
        """
        for plugin in self.plugins:
            if plugin.enabled:
                state = plugin.on_batch_start(state)
        return state
    
    def process_batch_end(self, state: TrainingState) -> None:
        """Process all plugins after batch."""
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_batch_end(state)
    
    def process_consolidate(self, state: TrainingState) -> None:
        """Process all plugins on consolidation."""
        for plugin in self.plugins:
            if plugin.enabled:
                plugin.on_consolidate(state)
    
    def reset_all(self) -> None:
        """Reset all plugins to initial state."""
        for plugin in self.plugins:
            plugin.reset()
    
    def get_state(self) -> Dict[str, Any]:
        """Get state of all plugins."""
        return {
            'num_plugins': len(self.plugins),
            'plugins': [p.get_state() for p in self.plugins],
        }
    
    def __repr__(self):
        plugin_names = [p.name for p in self.plugins]
        return f"PluginHost(plugins={plugin_names})"
    
    def __len__(self):
        return len(self.plugins)


# Example: Passthrough plugin (does nothing, for testing)
class PassthroughPlugin(LearningPlugin):
    """A plugin that does nothing. Useful for testing."""
    
    def __init__(self, name: str = "passthrough"):
        super().__init__(name)
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        return state


# Example: Logger plugin
class LoggerPlugin(LearningPlugin):
    """
    Logs metrics at specified intervals.
    
    This is a simple example of a plugin that doesn't modify hyperparameters,
    but reacts to metrics.
    """
    
    def __init__(self, name: str = "logger", log_interval: int = 10):
        super().__init__(name)
        self.log_interval = log_interval
        self.last_log_iter = 0
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        return state  # No modifications
    
    def on_batch_end(self, state: TrainingState) -> None:
        if state.iter_num - self.last_log_iter >= self.log_interval:
            print(f"[{self.name}] iter={state.iter_num}: "
                  f"loss={state.task_loss:.4f}, "
                  f"complexity={state.complexity_cost:.4f}, "
                  f"lr={state.learning_rate:.6f}")
            self.last_log_iter = state.iter_num
    
    def reset(self) -> None:
        self.last_log_iter = 0


if __name__ == "__main__":
    # Demo: Show plugin system
    print("=" * 80)
    print("PLUGIN SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create plugins
    logger = LoggerPlugin(name="logger", log_interval=5)
    passthrough = PassthroughPlugin(name="passthrough")
    
    # Create host
    host = PluginHost(plugins=[passthrough, logger])
    
    print(f"\nPlugin host: {host}")
    print(f"Number of plugins: {len(host)}")
    print("\nSimulating training loop...")
    print("-" * 80)
    
    # Simulate training
    state = TrainingState(
        learning_rate=1e-3,
        lambda_ockham=0.01,
        surprise_threshold=2.0,
    )
    
    for i in range(20):
        state.iter_num = i
        state.task_loss = 2.0 - i * 0.05  # Decreasing loss
        state.complexity_cost = 0.01 + i * 0.005  # Increasing complexity
        state.updated = True
        
        # Process batch start (plugins can modify hyperparameters)
        state = host.process_batch_start(state)
        
        # Training step would happen here (not shown)
        
        # Process batch end (plugins react to metrics)
        host.process_batch_end(state)
    
    print("-" * 80)
    print("\n✓ Demo complete!")
    print(f"\nFinal host state: {host.get_state()}")
