"""
demo_ockham.py - Simple demonstration of OckhamLearner

This script provides a minimal, self-contained example of how the OckhamLearner
works, without requiring a full training setup.

Usage:
    python demo_ockham.py
"""

import torch
import torch.nn as nn
from ockham_learner import OckhamLearner


# Create a simple toy model
class ToyModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("="*80)
    print("OCKHAM LEARNER DEMONSTRATION")
    print("="*80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    model = ToyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    # Create OckhamLearner with surprise gate
    learner = OckhamLearner(
        model=model,
        optimizer=optimizer,
        lambda_ockham=0.1,
        surprise_threshold=0.5,  # Only update if loss > 0.5
        device=device
    )
    
    print(f"\nOckham Configuration:")
    print(f"  Lambda: {learner.lambda_ockham}")
    print(f"  Surprise Threshold: {learner.surprise_threshold}")
    
    # Define loss function
    def loss_fn(logits, targets):
        return nn.functional.cross_entropy(logits, targets)
    
    # Generate some toy data
    print("\n" + "-"*80)
    print("Simulating adaptation on toy data...")
    print("-"*80)
    
    for i in range(20):
        # Create random input and target
        x = torch.randn(32, 10).to(device)
        y = torch.randint(0, 2, (32,)).to(device)
        
        # Adapt
        metrics = learner.adapt(x, y, loss_fn)
        
        # Print results
        status = "✓ UPDATED" if metrics['updated'] else "✗ SKIPPED"
        print(f"Batch {i+1:2d}: task_loss={metrics['task_loss']:.4f}, "
              f"complexity={metrics['complexity_cost']:.4f}, "
              f"total={metrics['total_loss']:.4f} [{status}]")
        
        # Consolidate after 10 batches
        if i == 9:
            print("\n>>> Consolidating anchor <<<\n")
            learner.consolidate()
    
    # Print final metrics
    print("\n" + "="*80)
    print("FINAL METRICS")
    print("="*80)
    metrics = learner.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    print("\n✓ Demo complete!")


if __name__ == '__main__':
    main()
