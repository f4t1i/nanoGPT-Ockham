"""
OccamContext: Standardized data object for plugin communication

This is the "VST-Rack" pattern - a standardized context object that flows
through the plugin chain, separating mechanics (code) from policy (config).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch


@dataclass
class OccamContext:
    """
    Standardized context object passed through plugin chain.
    
    This separates the "mechanics" (what the system CAN do) from the
    "policy" (what the system SHOULD do, configured via plugins).
    
    Design Principles:
    1. Immutable core data (logits, labels) - plugins read but don't modify
    2. Mutable hyperparameters - plugins can adjust these
    3. Metrics tracking - plugins can add custom metrics
    4. Anchor state - for TTT and knowledge preservation
    """
    
    # ===== IMMUTABLE CORE DATA =====
    # Plugins should read these but not modify them
    
    iter_num: int
    """Current training iteration"""
    
    task_loss: float
    """Current batch loss (task-specific)"""
    
    # ===== MUTABLE HYPERPARAMETERS =====
    # Plugins can modify these to control training
    
    learning_rate: float = 1e-3
    """Current learning rate"""
    
    lambda_ockham: float = 0.01
    """Anchor regularization strength"""
    
    surprise_threshold: float = 2.0
    """Minimum loss to trigger update"""
    
    consolidate_interval: int = 1000
    """Steps between anchor consolidations"""
    
    # ===== METRICS & STATE =====
    
    complexity_cost: float = 0.0
    """L2 distance from anchor (drift metric)"""
    
    grad_norm: float = 0.0
    """Gradient norm (for monitoring)"""
    
    update_rate: float = 1.0
    """Fraction of batches that triggered updates"""
    
    updated: bool = True
    """Whether this batch triggered an update"""
    
    consolidating: bool = False
    """Whether consolidation should happen this step"""
    
    # ===== ANCHOR STATE (for TTT) =====
    
    anchor_available: bool = False
    """Whether anchor state is available"""
    
    # ===== CUSTOM DATA =====
    # Plugins can store arbitrary data here
    
    custom: Dict[str, Any] = field(default_factory=dict)
    """Plugin-specific custom data"""
    
    # ===== OCCAM QUOTIENT (OQ) =====
    
    model_params: int = 0
    """Total number of model parameters"""
    
    active_params: int = 0
    """Number of non-zero parameters (for sparsity tracking)"""
    
    inference_flops: float = 0.0
    """Estimated FLOPs for inference (if available)"""
    
    memory_footprint_mb: float = 0.0
    """Model memory footprint in MB"""
    
    def compute_occam_quotient(self, task_performance: float) -> float:
        """
        Compute Occam Quotient (OQ):
        
        OQ = Task Performance / (Inference Cost × Memory Footprint)
        
        Higher is better - we want high performance with low cost.
        
        Args:
            task_performance: Task-specific metric (e.g., accuracy, 1/loss)
        
        Returns:
            Occam Quotient (higher = better efficiency)
        """
        # Use FLOPs if available, otherwise use param count as proxy
        cost_proxy = self.inference_flops if self.inference_flops > 0 else self.model_params
        
        # Use memory footprint if available, otherwise estimate from params
        memory_proxy = self.memory_footprint_mb if self.memory_footprint_mb > 0 else (self.model_params * 4 / 1e6)  # 4 bytes per param
        
        # Avoid division by zero
        if cost_proxy == 0 or memory_proxy == 0:
            return 0.0
        
        oq = task_performance / (cost_proxy * memory_proxy)
        return oq
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'iter_num': self.iter_num,
            'task_loss': self.task_loss,
            'learning_rate': self.learning_rate,
            'lambda_ockham': self.lambda_ockham,
            'surprise_threshold': self.surprise_threshold,
            'complexity_cost': self.complexity_cost,
            'grad_norm': self.grad_norm,
            'update_rate': self.update_rate,
            'updated': self.updated,
            'consolidating': self.consolidating,
            'model_params': self.model_params,
            'active_params': self.active_params,
            'inference_flops': self.inference_flops,
            'memory_footprint_mb': self.memory_footprint_mb,
        }
    
    def __repr__(self):
        return (
            f"OccamContext(iter={self.iter_num}, "
            f"loss={self.task_loss:.4f}, "
            f"lr={self.learning_rate:.6f}, "
            f"λ={self.lambda_ockham:.4f}, "
            f"updated={self.updated})"
        )


def create_context_from_training_state(state: 'TrainingState') -> OccamContext:
    """
    Convert old TrainingState to new OccamContext.
    
    This is a compatibility layer for migrating existing code.
    """
    return OccamContext(
        iter_num=state.iter_num,
        task_loss=state.task_loss,
        learning_rate=state.learning_rate,
        lambda_ockham=state.lambda_ockham,
        surprise_threshold=state.surprise_threshold,
        consolidate_interval=state.consolidate_interval,
        complexity_cost=state.complexity_cost,
        grad_norm=state.grad_norm,
        update_rate=state.update_rate,
        updated=state.updated,
        consolidating=state.consolidating,
        custom=state.custom.copy(),
    )


if __name__ == "__main__":
    print("=" * 80)
    print("OCKHAM CONTEXT DEMONSTRATION")
    print("=" * 80)
    
    # Create context
    ctx = OccamContext(
        iter_num=100,
        task_loss=1.5,
        learning_rate=1e-3,
        lambda_ockham=0.01,
        model_params=7_000_000_000,  # 7B params
        memory_footprint_mb=14_000,  # 14 GB
        inference_flops=1.4e12,  # 1.4 TFLOPs
    )
    
    print(f"\nContext: {ctx}")
    print(f"\nContext dict: {ctx.to_dict()}")
    
    # Compute Occam Quotient
    task_performance = 1.0 / ctx.task_loss  # Higher is better
    oq = ctx.compute_occam_quotient(task_performance)
    
    print(f"\nOccam Quotient Calculation:")
    print(f"  Task Performance: {task_performance:.4f} (1/loss)")
    print(f"  Inference FLOPs: {ctx.inference_flops:.2e}")
    print(f"  Memory Footprint: {ctx.memory_footprint_mb:.0f} MB")
    print(f"  Occam Quotient (OQ): {oq:.2e}")
    
    # Compare two models
    print("\n" + "-" * 80)
    print("COMPARING TWO MODELS")
    print("-" * 80)
    
    # Model A: Large, better performance
    ctx_a = OccamContext(
        iter_num=100,
        task_loss=1.2,
        model_params=13_000_000_000,  # 13B
        memory_footprint_mb=26_000,
        inference_flops=2.6e12,
    )
    perf_a = 1.0 / ctx_a.task_loss
    oq_a = ctx_a.compute_occam_quotient(perf_a)
    
    # Model B: Smaller, slightly worse performance
    ctx_b = OccamContext(
        iter_num=100,
        task_loss=1.3,
        model_params=7_000_000_000,  # 7B
        memory_footprint_mb=14_000,
        inference_flops=1.4e12,
    )
    perf_b = 1.0 / ctx_b.task_loss
    oq_b = ctx_b.compute_occam_quotient(perf_b)
    
    print(f"\nModel A (13B):")
    print(f"  Loss: {ctx_a.task_loss:.4f}")
    print(f"  Performance: {perf_a:.4f}")
    print(f"  OQ: {oq_a:.2e}")
    
    print(f"\nModel B (7B):")
    print(f"  Loss: {ctx_b.task_loss:.4f}")
    print(f"  Performance: {perf_b:.4f}")
    print(f"  OQ: {oq_b:.2e}")
    
    if oq_b > oq_a:
        print(f"\n✓ Model B wins! Better efficiency despite slightly worse loss.")
        print(f"  OQ improvement: {(oq_b / oq_a - 1) * 100:.1f}%")
    else:
        print(f"\n✓ Model A wins! Performance gain justifies increased cost.")
        print(f"  OQ improvement: {(oq_a / oq_b - 1) * 100:.1f}%")
    
    print("\n" + "=" * 80)
    print("✓ Demo complete!")
