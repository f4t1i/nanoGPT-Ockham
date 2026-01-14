"""
Core Refactoring Integration Demo

Demonstrates the complete V2 architecture:
1. OccamContext - Standardized data object
2. Plugins V2 - Using OccamContext
3. OckhamMemory V2 - Decision logic with OQ

This shows how mechanics (code) and policy (config) are cleanly separated.
"""

import numpy as np
from ockham_context import OccamContext
from plugins_v2 import OckhamGatePlugin, CompressorPlugin, EQPlugin, LimiterPlugin, SaturationPlugin
from ockham_memory_v2 import OckhamMemoryV2


def simulate_training_step(iter_num: int, base_loss: float = 2.0) -> OccamContext:
    """
    Simulate a training step, returning a context with realistic metrics.
    
    Args:
        iter_num: Current iteration
        base_loss: Base loss value
    
    Returns:
        OccamContext with simulated metrics
    """
    # Simulate improving loss with noise
    noise = 0.2 * np.random.randn()
    improvement = iter_num * 0.01
    task_loss = max(0.5, base_loss - improvement + noise)
    
    # Simulate increasing complexity (drift from anchor)
    complexity_cost = 0.02 + iter_num * 0.002 + 0.01 * np.random.rand()
    
    # Simulate gradient norm
    grad_norm = 0.5 + 0.3 * np.random.randn()
    
    # Create context
    ctx = OccamContext(
        iter_num=iter_num,
        task_loss=task_loss,
        learning_rate=1e-3,
        lambda_ockham=0.01,
        surprise_threshold=1.5,
        complexity_cost=complexity_cost,
        grad_norm=abs(grad_norm),
        model_params=7_000_000_000,  # 7B params
        memory_footprint_mb=14_000,  # 14 GB
        inference_flops=1.4e12,  # 1.4 TFLOPs
    )
    
    return ctx


def main():
    print("=" * 80)
    print("CORE REFACTORING INTEGRATION DEMO")
    print("=" * 80)
    print("\nThis demonstrates the complete V2 architecture:")
    print("  1. OccamContext - Standardized data object")
    print("  2. Plugins V2 - Using OccamContext")
    print("  3. OckhamMemory V2 - Decision logic with OQ")
    print("\n" + "=" * 80)
    
    # ===== SETUP =====
    print("\n[SETUP] Initializing components...")
    
    # Create all 5 plugins
    gate = OckhamGatePlugin(surprise_threshold=1.5, adaptive=False)
    compressor = CompressorPlugin(threshold=0.1, ratio=2.0, attack=0.1, release=0.05)
    eq = EQPlugin(bands={"easy": 0.5, "medium": 1.0, "hard": 1.5})
    limiter = LimiterPlugin(complexity_ceiling=0.15, force_consolidate=True)
    saturation = SaturationPlugin(drive=0.01, noise_type='learning_rate', warmup_iters=10)
    
    plugins = [gate, compressor, eq, limiter, saturation]
    print(f"  Plugins: {[p.name for p in plugins]}")
    
    # Create memory
    memory = OckhamMemoryV2(
        quality_threshold=1.8,
        complexity_threshold=0.12,
        min_oq_improvement=0.05,
    )
    print(f"  Memory: {memory}")
    
    # ===== TRAINING SIMULATION =====
    print("\n[TRAINING] Simulating 50 iterations...")
    print("-" * 80)
    
    prev_snapshot = None
    
    for i in range(50):
        # 1. Generate context (simulates forward pass + metrics)
        ctx = simulate_training_step(i, base_loss=2.0)
        
        # 2. Process through plugin chain (modifies hyperparameters)
        for plugin in plugins:
            if plugin.enabled:
                ctx = plugin.on_batch_start(ctx)
        
        # 3. Decision: Should we accept this model state?
        accept, reason = memory.should_accept_update(ctx, prev_snapshot)
        
        # 4. If accepted, create snapshot (simulated - no actual model)
        if accept:
            task_perf = 1.0 / max(ctx.task_loss, 1e-6)
            oq = ctx.compute_occam_quotient(task_perf)
            
            from ockham_memory_v2 import OccamMetrics, ModelSnapshot
            
            metrics = OccamMetrics(
                loss=ctx.task_loss,
                complexity_cost=ctx.complexity_cost,
                model_params=ctx.model_params,
                memory_footprint_mb=ctx.memory_footprint_mb,
                inference_flops=ctx.inference_flops,
                occam_quotient=oq,
            )
            
            prev_snapshot = ModelSnapshot(
                snapshot_id=f"iter{i}",
                iter_num=i,
                metrics=metrics,
            )
        
        # 5. Process batch end (plugins update internal state)
        for plugin in plugins:
            if plugin.enabled:
                plugin.on_batch_end(ctx)
        
        # 6. Log every 10 iterations
        if i % 10 == 0 or accept:
            status = "✓ ACCEPT" if accept else "✗ REJECT"
            updated_str = "✓" if ctx.updated else "✗"
            compressing = ctx.custom.get('compressor_compressing', False)
            comp_str = "🔴" if compressing else "🟢"
            
            task_perf = 1.0 / max(ctx.task_loss, 1e-6)
            oq = ctx.compute_occam_quotient(task_perf)
            
            print(
                f"Iter {i:2d}: loss={ctx.task_loss:.2f} {updated_str}, "
                f"complexity={ctx.complexity_cost:.3f} {comp_str}, "
                f"OQ={oq:.2e} → {status}"
            )
            if accept or i % 10 == 0:
                print(f"         λ={ctx.lambda_ockham:.4f}, lr={ctx.learning_rate:.6f}")
                if not accept and i % 10 == 0:
                    print(f"         Reason: {reason}")
    
    print("-" * 80)
    
    # ===== RESULTS =====
    print("\n[RESULTS] Final statistics")
    print("=" * 80)
    
    print("\n1. Plugin States:")
    for plugin in plugins:
        state = plugin.get_state()
        print(f"   {plugin.name}:")
        for key, value in state.items():
            if key not in ['name', 'enabled']:
                print(f"     - {key}: {value}")
    
    print("\n2. Memory Statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float) and value < 1:
            print(f"   - {key}: {value:.1%}")
        else:
            print(f"   - {key}: {value}")
    
    print("\n3. Key Insights:")
    update_rate = gate.get_state()['update_rate']
    accept_rate = stats['accept_rate']
    
    print(f"   - Update Rate (Gate): {update_rate:.1%}")
    print(f"     → {(1-update_rate)*100:.0f}% of batches skipped (compute saved)")
    
    print(f"   - Accept Rate (Memory): {accept_rate:.1%}")
    print(f"     → Only {accept_rate*100:.0f}% of models saved (storage saved)")
    
    if stats['best_oq'] > 0:
        print(f"   - Best Occam Quotient: {stats['best_oq']:.2e}")
        print(f"     → This is the most efficient model found")
    
    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print("\nKey Takeaway:")
    print("  The V2 architecture cleanly separates:")
    print("  - Mechanics (OccamContext, Plugins, Memory)")
    print("  - Policy (thresholds, ratios, gates)")
    print("  This makes the system production-ready and persona-friendly.")
    print("=" * 80)


if __name__ == "__main__":
    main()
