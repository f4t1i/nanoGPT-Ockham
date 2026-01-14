"""
OckhamMemory V2: Intelligent Model Selection with Occam Quotient

Extended version with:
1. should_accept_update() - Decision logic for accepting new model states
2. Occam Quotient (OQ) integration - Efficiency-based model comparison
3. OccamContext compatibility - Works with new context system
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from ockham_context import OccamContext


@dataclass
class OccamMetrics:
    """
    Metrics for evaluating a model according to Ockham's Razor.
    
    Attributes:
        loss: Task loss (lower is better)
        complexity_cost: L2 distance from anchor (lower is better)
        model_params: Total number of parameters
        memory_footprint_mb: Memory footprint in MB
        inference_flops: Estimated FLOPs for inference
        occam_quotient: Efficiency metric (higher is better)
    """
    loss: float
    complexity_cost: float
    model_params: int
    memory_footprint_mb: float
    inference_flops: float
    occam_quotient: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelSnapshot:
    """
    Represents a snapshot of model state with metrics.
    
    Attributes:
        snapshot_id: Unique identifier
        iter_num: Training iteration
        metrics: OccamMetrics for this snapshot
        checkpoint_path: Path to saved weights (optional)
    """
    snapshot_id: str
    iter_num: int
    metrics: OccamMetrics
    checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'snapshot_id': self.snapshot_id,
            'iter_num': self.iter_num,
            'metrics': self.metrics.to_dict(),
            'checkpoint_path': self.checkpoint_path,
        }


class OckhamMemoryV2:
    """
    Meta-level model selection based on Ockham's Razor.
    
    Key Features:
    1. Maintains Pareto frontier of (loss, complexity) pairs
    2. Uses Occam Quotient (OQ) for efficiency comparison
    3. Decides whether to accept new model states
    4. Prevents unnecessary model updates
    """
    
    def __init__(
        self,
        storage_path: str = "./.ockham/checkpoints",
        quality_threshold: float = 0.5,
        complexity_threshold: float = 0.2,
        min_oq_improvement: float = 0.05,
    ):
        """
        Initialize OckhamMemory.
        
        Args:
            storage_path: Directory for checkpoints
            quality_threshold: Maximum acceptable loss
            complexity_threshold: Maximum acceptable complexity cost
            min_oq_improvement: Minimum OQ improvement to accept new model (5%)
        """
        self.storage_path = storage_path
        self.quality_threshold = quality_threshold
        self.complexity_threshold = complexity_threshold
        self.min_oq_improvement = min_oq_improvement
        
        # State
        self.best_snapshot: Optional[ModelSnapshot] = None
        self.current_anchor: Optional[Dict[str, torch.Tensor]] = None
        self.frontier: List[ModelSnapshot] = []
        
        # Statistics
        self.total_evaluations = 0
        self.accepted_updates = 0
        self.rejected_quality = 0
        self.rejected_complexity = 0
        self.rejected_oq = 0
        
        os.makedirs(storage_path, exist_ok=True)
        print(f"[OckhamMemoryV2] Initialized with storage: {storage_path}")
    
    def register_anchor(self, model: nn.Module):
        """
        Sets the reference point (theta_anchor) for TTT regularization.
        
        Args:
            model: PyTorch model to use as anchor
        """
        # Deepcopy to CPU to save VRAM
        self.current_anchor = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        print(f"[OckhamMemoryV2] Anchor registered. Keys: {len(self.current_anchor)}")
    
    def should_accept_update(
        self,
        ctx: OccamContext,
        prev_snapshot: Optional[ModelSnapshot] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether to accept a new model state.
        
        This is the core Ockham decision logic:
        1. Quality gate: loss must be below threshold
        2. Complexity gate: complexity_cost must be below threshold
        3. Efficiency gate: OQ must improve by min_oq_improvement
        
        Args:
            ctx: Current OccamContext with metrics
            prev_snapshot: Previous best snapshot (if any)
        
        Returns:
            (accept: bool, reason: str)
        """
        self.total_evaluations += 1
        
        # Gate 1: Quality
        if ctx.task_loss > self.quality_threshold:
            self.rejected_quality += 1
            return False, f"REJECTED_QUALITY (loss {ctx.task_loss:.4f} > {self.quality_threshold})"
        
        # Gate 2: Complexity
        if ctx.complexity_cost > self.complexity_threshold:
            self.rejected_complexity += 1
            return False, f"REJECTED_COMPLEXITY (cost {ctx.complexity_cost:.4f} > {self.complexity_threshold})"
        
        # Gate 3: First model or OQ improvement
        if prev_snapshot is None:
            # First valid model
            self.accepted_updates += 1
            return True, "ACCEPTED_FIRST_VALID"
        
        # Compute current OQ
        task_performance = 1.0 / max(ctx.task_loss, 1e-6)
        current_oq = ctx.compute_occam_quotient(task_performance)
        
        # Compare with previous OQ
        prev_oq = prev_snapshot.metrics.occam_quotient
        oq_improvement = (current_oq - prev_oq) / max(prev_oq, 1e-6)
        
        if oq_improvement >= self.min_oq_improvement:
            self.accepted_updates += 1
            return True, f"ACCEPTED_OQ_IMPROVEMENT ({oq_improvement*100:.1f}%)"
        else:
            self.rejected_oq += 1
            return False, f"REJECTED_OQ_INSUFFICIENT (improvement {oq_improvement*100:.1f}% < {self.min_oq_improvement*100:.1f}%)"
    
    def save_checkpoint(
        self,
        model: nn.Module,
        ctx: OccamContext,
        tag: str = "ockham_best",
    ) -> ModelSnapshot:
        """
        Save model checkpoint if it passes Ockham criteria.
        
        Args:
            model: PyTorch model to save
            ctx: Current OccamContext
            tag: Checkpoint tag
        
        Returns:
            ModelSnapshot for this checkpoint
        """
        # Create metrics
        task_performance = 1.0 / max(ctx.task_loss, 1e-6)
        oq = ctx.compute_occam_quotient(task_performance)
        
        metrics = OccamMetrics(
            loss=ctx.task_loss,
            complexity_cost=ctx.complexity_cost,
            model_params=ctx.model_params,
            memory_footprint_mb=ctx.memory_footprint_mb,
            inference_flops=ctx.inference_flops,
            occam_quotient=oq,
        )
        
        # Save checkpoint
        path = os.path.join(self.storage_path, f"{tag}_iter{ctx.iter_num}.pt")
        torch.save({
            'model_state': model.state_dict(),
            'metrics': metrics.to_dict(),
            'iter_num': ctx.iter_num,
            'context': ctx.to_dict(),
        }, path)
        
        # Create snapshot
        snapshot = ModelSnapshot(
            snapshot_id=f"{tag}_iter{ctx.iter_num}",
            iter_num=ctx.iter_num,
            metrics=metrics,
            checkpoint_path=path,
        )
        
        # Update best
        if self.best_snapshot is None or oq > self.best_snapshot.metrics.occam_quotient:
            self.best_snapshot = snapshot
            print(f"[OckhamMemoryV2] New best! OQ={oq:.2e}, Loss={ctx.task_loss:.4f}")
        
        # Add to frontier
        self.frontier.append(snapshot)
        
        print(f"[OckhamMemoryV2] Checkpoint saved: {path}")
        return snapshot
    
    def get_statistics(self) -> Dict:
        """Get decision statistics for logging."""
        accept_rate = 0.0
        if self.total_evaluations > 0:
            accept_rate = self.accepted_updates / self.total_evaluations
        
        return {
            'total_evaluations': self.total_evaluations,
            'accepted_updates': self.accepted_updates,
            'rejected_quality': self.rejected_quality,
            'rejected_complexity': self.rejected_complexity,
            'rejected_oq': self.rejected_oq,
            'accept_rate': accept_rate,
            'best_oq': self.best_snapshot.metrics.occam_quotient if self.best_snapshot else 0.0,
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"OckhamMemoryV2("
            f"evals={stats['total_evaluations']}, "
            f"accepted={stats['accepted_updates']}, "
            f"rate={stats['accept_rate']:.1%}, "
            f"best_oq={stats['best_oq']:.2e})"
        )


if __name__ == "__main__":
    print("=" * 80)
    print("OCKHAM MEMORY V2 DEMONSTRATION")
    print("=" * 80)
    
    # Create memory
    memory = OckhamMemoryV2(
        quality_threshold=2.0,
        complexity_threshold=0.15,
        min_oq_improvement=0.05,
    )
    
    print(f"\nMemory: {memory}")
    print("\nSimulating model evolution...")
    print("-" * 80)
    
    # Simulate training
    prev_snapshot = None
    
    for i in range(20):
        # Create context (simulating improving model)
        ctx = OccamContext(
            iter_num=i * 100,
            task_loss=2.5 - i * 0.1,  # Improving loss
            complexity_cost=0.05 + i * 0.005,  # Increasing complexity
            model_params=7_000_000_000,
            memory_footprint_mb=14_000,
            inference_flops=1.4e12,
        )
        
        # Decide whether to accept
        accept, reason = memory.should_accept_update(ctx, prev_snapshot)
        
        # Log every 5 iterations
        if i % 5 == 0:
            status = "✓ ACCEPT" if accept else "✗ REJECT"
            task_perf = 1.0 / ctx.task_loss
            oq = ctx.compute_occam_quotient(task_perf)
            
            print(
                f"Iter {i:2d}: loss={ctx.task_loss:.2f}, "
                f"complexity={ctx.complexity_cost:.3f}, "
                f"OQ={oq:.2e} → {status}"
            )
            print(f"         Reason: {reason}")
        
        # If accepted, create snapshot (simulated - no actual model)
        if accept:
            task_perf = 1.0 / ctx.task_loss
            oq = ctx.compute_occam_quotient(task_perf)
            
            metrics = OccamMetrics(
                loss=ctx.task_loss,
                complexity_cost=ctx.complexity_cost,
                model_params=ctx.model_params,
                memory_footprint_mb=ctx.memory_footprint_mb,
                inference_flops=ctx.inference_flops,
                occam_quotient=oq,
            )
            
            prev_snapshot = ModelSnapshot(
                snapshot_id=f"sim_iter{i}",
                iter_num=i * 100,
                metrics=metrics,
            )
    
    print("-" * 80)
    print("\n✓ Demo complete!")
    print(f"\nFinal memory state: {memory}")
    print(f"\nStatistics: {memory.get_statistics()}")
