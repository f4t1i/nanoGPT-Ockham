# Plugin System Documentation

**nanoGPT-Ockham Plugin System: A VST-Inspired Architecture for Training Control**

---

## Overview

The nanoGPT-Ockham plugin system is a modular, composable architecture for controlling neural network training dynamics. Inspired by Digital Audio Workstations (DAWs) and VST plugins, it allows you to chain together "learning plugins" that modify hyperparameters and react to metrics in real-time.

**Key Concept:** Just as audio plugins transform sound signals, learning plugins transform training dynamics.

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        PluginHost                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Plugin 1 │→ │ Plugin 2 │→ │ Plugin 3 │→ │ Plugin N │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  TrainingState
         (hyperparameters + metrics)
```

**1. `TrainingState`** (dataclass)
- Encapsulates current training state
- Contains hyperparameters (mutable by plugins)
- Contains metrics (read by plugins)
- Passed through plugin chain

**2. `LearningPlugin`** (abstract base class)
- Defines plugin interface
- Three hooks:
  - `on_batch_start(state)` → modify hyperparameters
  - `on_batch_end(state)` → react to metrics
  - `on_consolidate(state)` → handle consolidation events

**3. `PluginHost`**
- Manages plugin chain
- Executes plugins in order
- Provides lifecycle management (add, remove, reset)

---

## Available Plugins

### 1. OckhamGatePlugin

**Purpose:** Implements the surprise gate - skips updates when task_loss < threshold.

**Audio Analogy:** Noise gate (cuts signal below threshold)

**Parameters:**
- `surprise_threshold`: Minimum loss to trigger update (default: 2.0)
- `adaptive`: Whether to adapt threshold based on loss distribution (default: False)
- `adaptation_rate`: Speed of threshold adaptation (default: 0.01)

**Behavior:**
- If `task_loss < surprise_threshold`: Skip update (set `state.updated = False`)
- If `task_loss >= surprise_threshold`: Allow update (set `state.updated = True`)
- If adaptive: Adjust threshold to median loss * 1.2

**Use When:**
- You want to save compute by skipping unnecessary updates
- Data contains noise or easy examples

**Example:**
```python
gate = OckhamGatePlugin(surprise_threshold=2.0, adaptive=False)
```

---

### 2. CompressorPlugin

**Purpose:** Dynamically adjusts `lambda_ockham` and `learning_rate` based on `complexity_cost`.

**Audio Analogy:** Compressor (reduces dynamic range)

**Parameters:**
- `threshold`: Complexity cost threshold to trigger compression (default: 0.1)
- `ratio`: Compression ratio - how much to increase lambda (default: 2.0)
- `attack`: Speed of compression increase (default: 0.1)
- `release`: Speed of compression decrease (default: 0.05)
- `makeup_gain`: Learning rate boost when stable (default: 1.0)
- `min_lambda`, `max_lambda`: Bounds for lambda_ockham

**Behavior:**
- If `complexity_cost > threshold`:
  - **Attack:** Increase `lambda_ockham` by ratio
  - Decrease `learning_rate` (compression)
- If `complexity_cost <= threshold`:
  - **Release:** Decrease `lambda_ockham` back to base
  - Increase `learning_rate` (makeup gain)

**Use When:**
- Model is drifting too far from anchor (high complexity_cost)
- You want automatic regularization adjustment

**Example:**
```python
compressor = CompressorPlugin(
    threshold=0.1,
    ratio=2.0,
    attack=0.1,
    release=0.05,
)
```

---

### 3. EQPlugin

**Purpose:** Curriculum learning via difficulty-based data weighting.

**Audio Analogy:** Equalizer (boosts/cuts frequency bands)

**Parameters:**
- `bands`: Dictionary mapping difficulty levels to gain values
  - Example: `{"easy": 0.5, "medium": 1.0, "hard": 1.5}`
- `adaptation_rate`: How quickly to adjust band gains (default: 0.01)

**Behavior:**
- Classifies current batch as easy/medium/hard based on loss percentiles
- Applies corresponding gain to `learning_rate`
- Gradually focuses more on harder examples

**Use When:**
- You want curriculum learning (start easy, increase difficulty)
- Data has varying difficulty levels

**Example:**
```python
eq = EQPlugin(
    bands={"easy": 0.5, "medium": 1.0, "hard": 1.5},
    adaptation_rate=0.01,
)
```

---

### 4. LimiterPlugin

**Purpose:** Hard caps on `complexity_cost` and `grad_norm`.

**Audio Analogy:** Limiter (hard ceiling on signal level)

**Parameters:**
- `complexity_ceiling`: Maximum allowed complexity_cost (default: 0.2)
- `grad_norm_ceiling`: Maximum allowed gradient norm (default: 1.0)
- `force_consolidate`: Whether to force consolidation when ceiling is hit (default: True)

**Behavior:**
- If `complexity_cost > complexity_ceiling`:
  - Increment hit counter
  - If `force_consolidate`: Set `state.consolidating = True`
- If `grad_norm > grad_norm_ceiling`:
  - Increment hit counter (informational only)

**Use When:**
- You want to prevent catastrophic forgetting
- You need hard safety bounds

**Example:**
```python
limiter = LimiterPlugin(
    complexity_ceiling=0.2,
    grad_norm_ceiling=1.0,
    force_consolidate=True,
)
```

---

### 5. SaturationPlugin

**Purpose:** Controlled noise injection for exploration and robustness.

**Audio Analogy:** Saturation/Drive (adds harmonics)

**Parameters:**
- `drive`: Amount of noise to inject (default: 0.01)
- `noise_type`: Where to inject ('learning_rate', 'gradient', 'both')
- `warmup_iters`: Number of iterations before noise is fully active (default: 100)

**Behavior:**
- Gradually increases noise from 0 to `drive` over `warmup_iters`
- If `noise_type` includes 'learning_rate': Adds Gaussian noise to LR
- If `noise_type` includes 'gradient': Sets flag for training loop to add gradient noise

**Use When:**
- You want to improve robustness via noise injection
- Exploring new domains or handling distribution shifts

**Example:**
```python
saturation = SaturationPlugin(
    drive=0.01,
    noise_type='learning_rate',
    warmup_iters=100,
)
```

---

## Presets

Presets are YAML files that define a complete plugin chain and base hyperparameters.

### Available Presets

#### 1. `ockham_tight.yaml`

**Philosophy:** Strong regularization, minimal adaptation, maximum stability

**Use When:**
- Preserving knowledge is critical
- Avoiding drift is more important than fast adaptation
- Fine-tuning a pre-trained model on a small dataset

**Plugin Chain:**
1. OckhamGatePlugin (threshold=2.5, non-adaptive)
2. CompressorPlugin (threshold=0.08, ratio=3.0, tight attack/release)
3. LimiterPlugin (complexity_ceiling=0.1, force consolidate)
4. SaturationPlugin (drive=0.005, minimal noise)

**Hyperparameters:**
- `learning_rate`: 0.0005 (low)
- `lambda_ockham`: 0.05 (high regularization)
- `surprise_threshold`: 2.5 (high - skip many updates)
- `consolidate_interval`: 500 (frequent consolidation)

---

#### 2. `balanced.yaml`

**Philosophy:** Default balanced configuration for most use cases

**Use When:**
- General-purpose training
- Unsure which preset to use
- Starting point for experimentation

**Plugin Chain:**
1. OckhamGatePlugin (threshold=2.0, non-adaptive)
2. CompressorPlugin (threshold=0.1, ratio=2.0, balanced attack/release)
3. EQPlugin (easy=0.5, medium=1.0, hard=1.5)
4. LimiterPlugin (complexity_ceiling=0.2, force consolidate)
5. SaturationPlugin (drive=0.01, moderate noise)

**Hyperparameters:**
- `learning_rate`: 0.001 (moderate)
- `lambda_ockham`: 0.01 (moderate regularization)
- `surprise_threshold`: 2.0 (moderate)
- `consolidate_interval`: 1000 (moderate)

---

#### 3. `exploratory.yaml`

**Philosophy:** High plasticity, aggressive adaptation, more exploration

**Use When:**
- Exploring new domains
- Handling large distribution shifts
- Test-time training on out-of-distribution data
- Fast adaptation is more important than stability

**Plugin Chain:**
1. OckhamGatePlugin (threshold=1.0, adaptive)
2. CompressorPlugin (threshold=0.15, ratio=1.5, loose attack/release, high makeup gain)
3. EQPlugin (easy=0.3, medium=1.0, hard=2.0 - focus on hard examples)
4. LimiterPlugin (complexity_ceiling=0.5, no forced consolidation)
5. SaturationPlugin (drive=0.03, both LR and gradient noise)

**Hyperparameters:**
- `learning_rate`: 0.002 (high)
- `lambda_ockham`: 0.001 (low regularization)
- `surprise_threshold`: 1.0 (low - update frequently)
- `consolidate_interval`: 2000 (infrequent consolidation)

---

## Usage

### Basic Usage

```python
from preset_loader import load_preset_simple
from plugin_system import TrainingState

# Load preset
host = load_preset_simple("balanced")

# Initialize state
state = TrainingState(
    iter_num=0,
    learning_rate=0.001,
    lambda_ockham=0.01,
    surprise_threshold=2.0,
    task_loss=2.5,
    complexity_cost=0.08,
    grad_norm=0.5,
)

# Process batch start (plugins modify hyperparameters)
state = host.process_batch_start(state)

# Training step happens here (not shown)
# ...

# Process batch end (plugins react to metrics)
host.process_batch_end(state)
```

### Advanced Usage: Custom Plugin Chain

```python
from plugin_system import PluginHost
from plugins import OckhamGatePlugin, CompressorPlugin, LimiterPlugin

# Create custom plugin chain
gate = OckhamGatePlugin(surprise_threshold=1.5)
compressor = CompressorPlugin(threshold=0.12, ratio=2.5)
limiter = LimiterPlugin(complexity_ceiling=0.15)

host = PluginHost(plugins=[gate, compressor, limiter])

# Use as above
```

### Creating Custom Plugins

```python
from plugin_system import LearningPlugin, TrainingState

class MyCustomPlugin(LearningPlugin):
    def __init__(self, name="my_plugin", my_param=1.0):
        super().__init__(name)
        self.my_param = my_param
    
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        # Modify hyperparameters based on metrics
        if state.task_loss > 3.0:
            state.learning_rate *= 0.5  # Reduce LR if loss is high
        return state
    
    def on_batch_end(self, state: TrainingState) -> None:
        # React to metrics (e.g., logging)
        if state.updated:
            print(f"[{self.name}] Updated at iter {state.iter_num}")
    
    def reset(self) -> None:
        # Reset internal state
        pass
```

---

## Design Principles

### 1. Composability

Plugins are independent and can be chained in any order. Each plugin receives the state modified by previous plugins.

### 2. Transparency

All plugin behavior is explicit and inspectable. Use `plugin.get_state()` to see internal state.

### 3. Simplicity

Each plugin does one thing well. Complex behaviors emerge from plugin combinations.

### 4. Ockham's Razor

Start with minimal plugins. Add complexity only when needed.

---

## Comparison with Existing Approaches

| Approach | Modularity | Composability | Interpretability | Ease of Use |
|:---|:---:|:---:|:---:|:---:|
| **Manual Hyperparameter Tuning** | ✗ | ✗ | ✓ | ✗ |
| **Keras Callbacks** | ✓ | ~ | ~ | ✓ |
| **PyTorch Lightning Callbacks** | ✓ | ~ | ~ | ✓ |
| **AutoML (Optuna, Ray Tune)** | ✗ | ✗ | ✗ | ✓ |
| **RL-based Meta-Optimizers** | ✗ | ✗ | ✗ | ✗ |
| **nanoGPT-Ockham Plugins** | ✓✓ | ✓✓ | ✓✓ | ✓✓ |

**Key Differentiators:**
- **Presets:** One-line behavior switching (like VST presets)
- **DAW Metaphor:** Familiar mental model for audio engineers and creative coders
- **Explicit State:** No hidden magic, all behavior is visible

---

## Future Extensions

### Short-term (1-2 months)
- Integration with `train_ockham.py` for seamless training
- Visualization dashboard for real-time plugin monitoring
- More presets for specific domains (vision, NLP, RL)

### Medium-term (3-6 months)
- Automatic preset selection via meta-learning
- Plugin marketplace for community contributions
- GUI for preset design (web-based or desktop)

### Long-term (6+ months)
- Learned plugins (plugins that adapt their own parameters)
- Multi-task plugin memory (share plugin states across tasks)
- Integration with Weights & Biases, TensorBoard

---

## FAQ

**Q: Do plugins slow down training?**  
A: Minimal overhead. Plugins are pure Python functions with no heavy computation. Typical overhead: <1% of training time.

**Q: Can I use plugins without presets?**  
A: Yes! Create a `PluginHost` manually and add plugins directly.

**Q: Can plugins conflict with each other?**  
A: Plugins are executed in order. Later plugins see the state modified by earlier plugins. Design your chain carefully.

**Q: How do I debug plugin behavior?**  
A: Use `plugin.get_state()` to inspect internal state. Add `LoggerPlugin` to your chain for automatic logging.

**Q: Can I use this with other frameworks (PyTorch Lightning, Hugging Face)?**  
A: Yes, but you'll need to integrate the plugin system manually. The core architecture is framework-agnostic.

---

## References

- **Ockham's Razor:** "Entities should not be multiplied beyond necessity"
- **Audio Compression:** [Wikipedia: Dynamic Range Compression](https://en.wikipedia.org/wiki/Dynamic_range_compression)
- **VST Plugins:** [Wikipedia: Virtual Studio Technology](https://en.wikipedia.org/wiki/Virtual_Studio_Technology)
- **Curriculum Learning:** Bengio et al., "Curriculum Learning" (ICML 2009)
- **Elastic Weight Consolidation:** Kirkpatrick et al., "Overcoming catastrophic forgetting" (PNAS 2017)

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-14  
**Author:** nanoGPT-Ockham Project
