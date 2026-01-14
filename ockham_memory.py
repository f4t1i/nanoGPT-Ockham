"""
OckhamMemory: Intelligent Model Selection via Ockham's Razor

This module implements the OckhamMemory class, which maintains a "memory" of
trained models and selects the simplest model that meets performance criteria.

Key Principle:
"Choose the simplest model that solves the problem adequately."

This operationalizes Ockham's Razor at the architecture level, ensuring that
we don't use unnecessarily complex models when simpler ones suffice.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ModelCandidate:
    """
    Represents a candidate model in the Ockham selection process.
    
    Attributes:
        model_id: Unique identifier for this model
        n_params: Total number of parameters
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        val_loss: Validation loss achieved
        train_loss: Training loss achieved
        checkpoint_path: Path to saved model weights
        config: Full model configuration dict
    """
    model_id: str
    n_params: int
    n_layer: int
    n_head: int
    n_embd: int
    val_loss: float
    train_loss: float
    checkpoint_path: str
    config: Dict


class OckhamMemory:
    """
    Maintains a Pareto frontier of models: the set of models where no other
    model is both simpler AND better performing.
    
    The "best" model according to Ockham's Razor is the simplest model on
    this frontier that meets a specified performance threshold.
    """
    
    def __init__(self, memory_dir: str = 'ockham_memory'):
        """
        Initialize the OckhamMemory.
        
        Args:
            memory_dir: Directory to store model metadata and checkpoints
        """
        self.memory_dir = memory_dir
        self.frontier_path = os.path.join(memory_dir, 'frontier.json')
        self.candidates: List[ModelCandidate] = []
        
        # Create directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # Load existing frontier if available
        self._load_frontier()
    
    def _load_frontier(self):
        """Load the Pareto frontier from disk if it exists."""
        if os.path.exists(self.frontier_path):
            with open(self.frontier_path, 'r') as f:
                data = json.load(f)
                self.candidates = [ModelCandidate(**item) for item in data]
            print(f"[OckhamMemory] Loaded {len(self.candidates)} candidates from {self.frontier_path}")
        else:
            print(f"[OckhamMemory] No existing frontier found. Starting fresh.")
    
    def _save_frontier(self):
        """Save the current Pareto frontier to disk."""
        with open(self.frontier_path, 'w') as f:
            data = [asdict(candidate) for candidate in self.candidates]
            json.dump(data, f, indent=2)
        print(f"[OckhamMemory] Saved {len(self.candidates)} candidates to {self.frontier_path}")
    
    def _is_dominated(self, candidate: ModelCandidate) -> bool:
        """
        Check if a candidate is dominated by any existing candidate.
        
        A candidate is dominated if there exists another candidate that is:
        - Simpler (fewer parameters) AND better (lower val_loss)
        OR
        - Equally simple AND better
        OR
        - Simpler AND equally good
        
        Args:
            candidate: The candidate to check
            
        Returns:
            True if the candidate is dominated, False otherwise
        """
        for existing in self.candidates:
            simpler = existing.n_params < candidate.n_params
            equally_simple = existing.n_params == candidate.n_params
            better = existing.val_loss < candidate.val_loss
            equally_good = abs(existing.val_loss - candidate.val_loss) < 1e-6
            
            if (simpler and (better or equally_good)) or (equally_simple and better):
                return True
        
        return False
    
    def _remove_dominated(self, new_candidate: ModelCandidate):
        """
        Remove any existing candidates that are dominated by the new candidate.
        
        Args:
            new_candidate: The new candidate that may dominate existing ones
        """
        self.candidates = [
            existing for existing in self.candidates
            if not (
                (new_candidate.n_params < existing.n_params and 
                 new_candidate.val_loss <= existing.val_loss) or
                (new_candidate.n_params <= existing.n_params and 
                 new_candidate.val_loss < existing.val_loss)
            )
        ]
    
    def add_candidate(
        self,
        model: nn.Module,
        val_loss: float,
        train_loss: float,
        config: Dict,
        model_id: Optional[str] = None
    ) -> bool:
        """
        Add a new model candidate to the memory.
        
        The candidate will only be added if it's not dominated by existing
        candidates. If added, it may cause some existing candidates to be
        removed if they're now dominated.
        
        Args:
            model: The trained PyTorch model
            val_loss: Validation loss achieved by this model
            train_loss: Training loss achieved by this model
            config: Configuration dict containing model architecture details
            model_id: Optional unique identifier (auto-generated if not provided)
            
        Returns:
            True if the candidate was added to the frontier, False if dominated
        """
        # Generate model_id if not provided
        if model_id is None:
            model_id = f"model_{len(self.candidates):04d}"
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        
        # Extract architecture details from config
        n_layer = config.get('n_layer', 0)
        n_head = config.get('n_head', 0)
        n_embd = config.get('n_embd', 0)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(self.memory_dir, f"{model_id}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'val_loss': val_loss,
            'train_loss': train_loss,
        }, checkpoint_path)
        
        # Create candidate
        candidate = ModelCandidate(
            model_id=model_id,
            n_params=n_params,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            val_loss=val_loss,
            train_loss=train_loss,
            checkpoint_path=checkpoint_path,
            config=config
        )
        
        # Check if dominated
        if self._is_dominated(candidate):
            print(f"[OckhamMemory] Candidate {model_id} is dominated. Not adding to frontier.")
            # Remove the checkpoint since we're not keeping this candidate
            os.remove(checkpoint_path)
            return False
        
        # Remove any candidates that are now dominated by this one
        self._remove_dominated(candidate)
        
        # Add to frontier
        self.candidates.append(candidate)
        
        # Sort by complexity (ascending)
        self.candidates.sort(key=lambda c: c.n_params)
        
        # Save updated frontier
        self._save_frontier()
        
        print(f"[OckhamMemory] Added candidate {model_id} to frontier "
              f"(params={n_params:,}, val_loss={val_loss:.4f})")
        
        return True
    
    def get_best_model(
        self,
        max_val_loss: Optional[float] = None,
        max_params: Optional[int] = None
    ) -> Optional[ModelCandidate]:
        """
        Get the best model according to Ockham's Razor.
        
        "Best" means: the simplest model that meets the specified constraints.
        
        Args:
            max_val_loss: Maximum acceptable validation loss (default: None, no constraint)
            max_params: Maximum acceptable parameter count (default: None, no constraint)
            
        Returns:
            The ModelCandidate that is simplest among those meeting constraints,
            or None if no candidate meets the constraints
        """
        # Filter candidates by constraints
        valid_candidates = self.candidates
        
        if max_val_loss is not None:
            valid_candidates = [c for c in valid_candidates if c.val_loss <= max_val_loss]
        
        if max_params is not None:
            valid_candidates = [c for c in valid_candidates if c.n_params <= max_params]
        
        if not valid_candidates:
            print(f"[OckhamMemory] No candidates meet the specified constraints.")
            return None
        
        # Return the simplest (already sorted by n_params)
        best = valid_candidates[0]
        print(f"[OckhamMemory] Best model: {best.model_id} "
              f"(params={best.n_params:,}, val_loss={best.val_loss:.4f})")
        return best
    
    def load_best_model(
        self,
        model: nn.Module,
        max_val_loss: Optional[float] = None,
        max_params: Optional[int] = None
    ) -> bool:
        """
        Load the weights of the best model into the provided model instance.
        
        Args:
            model: The model instance to load weights into
            max_val_loss: Maximum acceptable validation loss
            max_params: Maximum acceptable parameter count
            
        Returns:
            True if a model was found and loaded, False otherwise
        """
        best = self.get_best_model(max_val_loss=max_val_loss, max_params=max_params)
        
        if best is None:
            return False
        
        checkpoint = torch.load(best.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[OckhamMemory] Loaded weights from {best.checkpoint_path}")
        return True
    
    def get_frontier_summary(self) -> str:
        """
        Get a human-readable summary of the current Pareto frontier.
        
        Returns:
            Formatted string describing all candidates on the frontier
        """
        if not self.candidates:
            return "[OckhamMemory] Frontier is empty."
        
        lines = ["[OckhamMemory] Current Pareto Frontier:"]
        lines.append("-" * 80)
        lines.append(f"{'Model ID':<15} {'Params':>12} {'Layers':>7} {'Heads':>6} {'Embd':>6} {'Val Loss':>10} {'Train Loss':>11}")
        lines.append("-" * 80)
        
        for candidate in self.candidates:
            lines.append(
                f"{candidate.model_id:<15} "
                f"{candidate.n_params:>12,} "
                f"{candidate.n_layer:>7} "
                f"{candidate.n_head:>6} "
                f"{candidate.n_embd:>6} "
                f"{candidate.val_loss:>10.4f} "
                f"{candidate.train_loss:>11.4f}"
            )
        
        lines.append("-" * 80)
        return "\n".join(lines)
    
    def clear(self):
        """
        Clear all candidates and remove all checkpoints.
        
        WARNING: This is destructive and cannot be undone!
        """
        # Remove all checkpoint files
        for candidate in self.candidates:
            if os.path.exists(candidate.checkpoint_path):
                os.remove(candidate.checkpoint_path)
        
        # Clear candidates list
        self.candidates = []
        
        # Save empty frontier
        self._save_frontier()
        
        print(f"[OckhamMemory] Cleared all candidates and checkpoints.")
