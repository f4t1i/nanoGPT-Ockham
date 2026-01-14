"""
Demo: Plugin System with Presets

Shows how to use the plugin system with different presets.
"""

import numpy as np
from plugin_system import TrainingState, PluginHost
from preset_loader import PresetLoader


def simulate_training(preset_name: str, num_iters: int = 50):
    """Simulate training with a given preset."""
    print("=" * 80)
    print(f"SIMULATING TRAINING WITH PRESET: {preset_name.upper()}")
    print("=" * 80)
    
    # Load preset
    loader = PresetLoader(presets_dir="presets")
    host = loader.create_plugin_host(preset_name)
    hparams = loader.get_hyperparameters(preset_name)
    info = loader.get_preset_info(preset_name)
    
    print(f"\nPreset: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"\nPlugin chain ({len(host)} plugins):")
    for i, plugin in enumerate(host.plugins, 1):
        print(f"  {i}. {plugin.name} ({plugin.__class__.__name__})")
    
    print(f"\nBase hyperparameters:")
    for key, value in hparams.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "-" * 80)
    print("Training simulation...")
    print("-" * 80)
    
    # Initialize state
    state = TrainingState(
        learning_rate=hparams['learning_rate'],
        lambda_ockham=hparams['lambda_ockham'],
        surprise_threshold=hparams['surprise_threshold'],
    )
    
    # Simulate training
    for i in range(num_iters):
        state.iter_num = i
        
        # Simulate varying metrics
        state.task_loss = 2.0 + 0.5 * np.sin(i * 0.2) - i * 0.01
        state.complexity_cost = 0.05 + i * 0.003
        state.grad_norm = 0.5 + 0.2 * np.random.randn()
        
        # Process batch
        state = host.process_batch_start(state)
        host.process_batch_end(state)
        
        # Log every 10 iterations
        if i % 10 == 0 or i == num_iters - 1:
            updated_str = "✓" if state.updated else "✗"
            print(
                f"Iter {i:3d}: loss={state.task_loss:5.2f} {updated_str}, "
                f"complexity={state.complexity_cost:.3f}, "
                f"λ={state.lambda_ockham:.4f}, "
                f"lr={state.learning_rate:.6f}"
            )
    
    print("-" * 80)
    print("\nFinal plugin states:")
    for plugin in host.plugins:
        print(f"\n{plugin.name}:")
        for key, value in plugin.get_state().items():
            if key not in ['name', 'enabled']:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    print("\n" * 2)
    print("=" * 80)
    print("PLUGIN SYSTEM DEMONSTRATION WITH PRESETS")
    print("=" * 80)
    print()
    
    # Test all three presets
    presets = ["ockham_tight", "balanced", "exploratory"]
    
    for preset in presets:
        simulate_training(preset, num_iters=30)
        print("\n" * 2)
    
    print("=" * 80)
    print("✓ All presets tested successfully!")
    print("=" * 80)
