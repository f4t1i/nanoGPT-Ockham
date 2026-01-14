# Future Directions: nanoGPT-Ockham

This document outlines potential future extensions and improvements to the nanoGPT-Ockham framework.

---

## 1. VST-Style Plugin System for Learning Methods

### Vision: "DAW for Learning"

Inspired by Digital Audio Workstations (DAWs) and VST plugins, we can create a **modular plugin architecture** for training methods. Just as audio plugins transform sound signals, **learning plugins** would transform training dynamics.

### Core Concept

**Current State:**
- Training = monolithic script with hardcoded hyperparameters
- Changing behavior = manually editing 10+ parameters

**Future State:**
- Training = plugin host with swappable modules
- Changing behavior = switching a preset (like in a DAW)

### Plugin Architecture

```python
class LearningPlugin:
    """Base class for all learning plugins."""
    
    def on_batch_start(self, state: TrainingState) -> HyperParams:
        """Called before processing a batch."""
        return state.hparams
    
    def on_batch_end(self, state: TrainingState, metrics: Dict) -> None:
        """Called after processing a batch."""
        pass
    
    def on_consolidate(self, state: TrainingState) -> None:
        """Called when model consolidates."""
        pass


class PluginHost:
    """Manages and chains learning plugins."""
    
    def __init__(self, plugins: List[LearningPlugin]):
        self.plugins = plugins
    
    def process_batch(self, batch, state):
        # Pre-processing: let plugins adjust hyperparameters
        for plugin in self.plugins:
            state.hparams = plugin.on_batch_start(state)
        
        # Execute training step
        metrics = train_step(batch, state.hparams)
        
        # Post-processing: let plugins react to metrics
        for plugin in self.plugins:
            plugin.on_batch_end(state, metrics)
        
        return metrics
```

---

## 2. Mapping: Audio Plugins → Learning Plugins

### Compressor → Learning Compressor

**Audio Compressor:**
- Threshold: Signal level that triggers compression
- Ratio: How much to reduce signal above threshold
- Attack/Release: How fast compression engages/disengages

**Learning Compressor:**
```python
class CompressorPlugin(LearningPlugin):
    """
    Compresses gradient updates based on loss magnitude.
    
    - Threshold: surprise_threshold (minimum loss to trigger update)
    - Ratio: How much to scale gradients when loss is high
    - Attack/Release: How quickly to adjust learning rate
    """
    
    def __init__(self, threshold=2.0, ratio=0.5, attack=0.1, release=0.01):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.current_scale = 1.0
    
    def on_batch_start(self, state):
        loss = state.metrics.get('task_loss', 0)
        
        # If loss exceeds threshold, compress (reduce) learning rate
        if loss > self.threshold:
            target_scale = self.ratio
            self.current_scale += self.attack * (target_scale - self.current_scale)
        else:
            self.current_scale += self.release * (1.0 - self.current_scale)
        
        state.hparams.learning_rate *= self.current_scale
        return state.hparams
```

**Use Case:** Automatically dampen learning when model is unstable (high loss), restore when stable.

---

### EQ → Curriculum/Loss EQ

**Audio EQ:**
- Multiple frequency bands (bass, mid, treble)
- Each band has gain control
- Shapes the overall frequency response

**Learning EQ:**
```python
class CurriculumEQPlugin(LearningPlugin):
    """
    Adjusts weights for different data "bands" (easy vs. hard samples).
    
    - Bands: Different difficulty levels or data domains
    - Gain per band: How much to emphasize each band
    - Q-factor: How selective the filtering is
    """
    
    def __init__(self, bands: Dict[str, float]):
        # bands = {"easy": 0.5, "medium": 1.0, "hard": 1.5}
        self.bands = bands
    
    def on_batch_start(self, state):
        # Classify current batch difficulty
        difficulty = self._estimate_difficulty(state.batch)
        
        # Apply gain for this band
        weight = self.bands.get(difficulty, 1.0)
        state.hparams.loss_weight = weight
        
        return state.hparams
    
    def _estimate_difficulty(self, batch):
        # Simple heuristic: use loss from previous epoch
        # Or: use pre-computed difficulty scores
        return "medium"  # Placeholder
```

**Use Case:** Implement curriculum learning by gradually increasing weight on harder examples.

---

### Limiter → Safety Layer

**Audio Limiter:**
- Hard ceiling on signal level
- Prevents clipping/distortion

**Learning Limiter:**
```python
class LimiterPlugin(LearningPlugin):
    """
    Hard cap on gradient norm, parameter change, or complexity cost.
    
    - Ceiling: Maximum allowed value
    - Type: What to limit (grad_norm, delta_theta, complexity_cost)
    """
    
    def __init__(self, ceiling=1.0, limit_type='grad_norm'):
        self.ceiling = ceiling
        self.limit_type = limit_type
    
    def on_batch_end(self, state, metrics):
        if self.limit_type == 'complexity_cost':
            if metrics['complexity_cost'] > self.ceiling:
                # Force consolidation to reset anchor
                state.learner.consolidate()
                print(f"[Limiter] Complexity exceeded {self.ceiling}, consolidating.")
```

**Use Case:** Prevent catastrophic forgetting by limiting how far model can drift from anchor.

---

### Saturation/Drive → Controlled Noise Injection

**Audio Saturation:**
- Adds harmonics and warmth
- Controlled distortion for character

**Learning Saturation:**
```python
class SaturationPlugin(LearningPlugin):
    """
    Adds controlled noise to gradients or embeddings.
    
    - Drive: Amount of noise to inject
    - Type: Where to inject (gradients, embeddings, learning rate)
    """
    
    def __init__(self, drive=0.01, noise_type='gradient'):
        self.drive = drive
        self.noise_type = noise_type
    
    def on_batch_end(self, state, metrics):
        if self.noise_type == 'gradient':
            for param in state.model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.drive
                    param.grad += noise
```

**Use Case:** Improve robustness and exploration, especially in test-time training.

---

## 3. Preset System

Instead of manually tuning 20 hyperparameters, users load a **preset** that configures the entire plugin chain.

### Example Presets

**`preset_occam_tight.yaml`**
```yaml
name: "Ockham Tight"
description: "Strong regularization, minimal adaptation, maximum stability"
plugins:
  - type: CompressorPlugin
    params:
      threshold: 2.5
      ratio: 0.3
      attack: 0.05
      release: 0.02
  
  - type: LimiterPlugin
    params:
      ceiling: 0.1
      limit_type: complexity_cost
  
  - type: OccamGatePlugin
    params:
      lambda_ockham: 0.05
      surprise_threshold: 2.0
```

**`preset_exploratory.yaml`**
```yaml
name: "Exploratory Mix"
description: "High plasticity, more noise, aggressive adaptation"
plugins:
  - type: CompressorPlugin
    params:
      threshold: 1.0
      ratio: 0.8
      attack: 0.2
      release: 0.1
  
  - type: SaturationPlugin
    params:
      drive: 0.02
      noise_type: gradient
  
  - type: OccamGatePlugin
    params:
      lambda_ockham: 0.001
      surprise_threshold: 0.5
```

**`preset_balanced.yaml`**
```yaml
name: "Balanced"
description: "Default balanced configuration"
plugins:
  - type: CompressorPlugin
    params:
      threshold: 2.0
      ratio: 0.5
      attack: 0.1
      release: 0.05
  
  - type: CurriculumEQPlugin
    params:
      bands:
        easy: 0.5
        medium: 1.0
        hard: 1.5
  
  - type: OccamGatePlugin
    params:
      lambda_ockham: 0.01
      surprise_threshold: 1.5
```

### Usage

```python
from plugin_host import PluginHost
from presets import load_preset

# Load preset
preset = load_preset("preset_occam_tight.yaml")

# Create plugin host
host = PluginHost(preset.plugins)

# Use in training
for batch in dataloader:
    metrics = host.process_batch(batch, state)
```

---

## 4. Macro Controls

Like synthesizers have "macro knobs" that control multiple parameters at once, we can create high-level controls:

### Stability ↔ Plasticity Slider

```python
def set_stability_plasticity(value: float):
    """
    value = 0.0: Maximum stability (tight Ockham, high lambda)
    value = 1.0: Maximum plasticity (loose Ockham, low lambda)
    """
    lambda_ockham = 0.1 * (1.0 - value) + 0.001 * value
    surprise_threshold = 3.0 * (1.0 - value) + 0.5 * value
    consolidate_interval = 1000 * (1.0 - value) + 100 * value
    
    return {
        'lambda_ockham': lambda_ockham,
        'surprise_threshold': surprise_threshold,
        'consolidate_interval': int(consolidate_interval)
    }
```

### Sparsity ↔ Accuracy Slider

```python
def set_sparsity_accuracy(value: float):
    """
    value = 0.0: Maximum sparsity (skip many updates, save compute)
    value = 1.0: Maximum accuracy (update on everything)
    """
    surprise_threshold = 5.0 * (1.0 - value) + 0.1 * value
    update_frequency = 0.2 * (1.0 - value) + 1.0 * value
    
    return {
        'surprise_threshold': surprise_threshold,
        'target_update_rate': update_frequency
    }
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Current)
- ✅ `OckhamLearner` with surprise gate
- ✅ `OckhamMemory` for model selection
- ✅ Basic training integration

### Phase 2: Plugin-Ready Refactor (2-4 weeks)
- Add internal hooks to `OckhamLearner`:
  - `on_batch_start()`
  - `on_adapt()`
  - `on_consolidate()`
- Make hyperparameters dynamically adjustable
- **No plugin system yet**, just the hooks

### Phase 3: Plugin System (1-2 months)
- Implement `PluginHost` and `LearningPlugin` base class
- Build 3 core plugins:
  - `CompressorPlugin`
  - `LimiterPlugin`
  - `CurriculumEQPlugin`
- Create 3 presets: Tight, Balanced, Exploratory
- YAML-based preset loading

### Phase 4: Expansion (3-6 months)
- Add more plugins:
  - `SaturationPlugin` (noise injection)
  - `DelayPlugin` (memory horizon)
  - `ReverbPlugin` (gradient echo)
- Macro control system
- Visualization dashboard (show plugin chain, real-time metrics)

### Phase 5: Community (6+ months)
- Plugin API documentation
- Community plugin repository
- GUI (web-based or desktop) for preset design
- Integration with Weights & Biases, TensorBoard

---

## 6. Why This Matters

### Current Problem
- Training is opaque and hard to control
- Hyperparameter tuning is trial-and-error
- Behavior changes require deep code modifications

### With Plugin System
- Training becomes **transparent** (see what each plugin does)
- Behavior changes via **presets** (no code changes)
- **Composable** (mix and match plugins)
- **Shareable** (distribute presets like VST presets)

### Philosophical Alignment
- **Ockham's Razor:** Start with minimal plugins, add only what's necessary
- **MIRAS:** Modular, Interpretable, Reproducible, Auditable, Stable
- **Engineering over Alchemy:** Explicit, measurable, controllable

---

## 7. Open Questions

1. **Plugin Ordering:** Does order matter? (Like audio FX chains)
2. **Plugin Conflicts:** What if two plugins try to set the same hyperparameter?
3. **Performance:** Does the plugin overhead slow down training?
4. **Preset Discovery:** How do users find the right preset for their task?
5. **Automatic Tuning:** Can we use RL/Bayesian optimization to tune plugin parameters?

---

## 8. Related Work

- **Keras Callbacks:** Similar concept, but not as modular or composable
- **PyTorch Lightning Callbacks:** More structured, but still code-heavy
- **AutoML (e.g., Optuna):** Tunes hyperparameters, but doesn't provide plugin-style modularity
- **RL for Hyperparameter Optimization:** Learns to adjust parameters, but lacks interpretability

**Our Differentiator:** The DAW/VST metaphor + preset system + explicit modularity.

---

## 9. Next Steps

**Immediate (after Ockham v1.0 is proven):**
1. Write detailed plugin API specification
2. Refactor `OckhamLearner` to support hooks
3. Implement `PluginHost` skeleton

**Short-term (1-2 months):**
1. Build 3 core plugins
2. Create 3 presets
3. Test on Shakespeare and TinyStories datasets

**Long-term (3-6 months):**
1. Expand plugin library
2. Build visualization dashboard
3. Open-source and invite community contributions

---

## 10. Call to Action

If you're reading this and find the idea compelling, here's how you can help:

1. **Try nanoGPT-Ockham** and give feedback
2. **Propose plugin ideas** (what would be useful for your use case?)
3. **Contribute code** (implement a plugin or preset)
4. **Share presets** (if you find a good configuration, share it!)

---

**"Entities should not be multiplied beyond necessity."**  
*Let's build training systems that respect this wisdom.*

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-14  
**Author:** nanoGPT-Ockham Project
