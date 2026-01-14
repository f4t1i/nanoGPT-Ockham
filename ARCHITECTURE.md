# nanoGPT-Ockham Architecture (V2)

**Core Refactoring: Separating Mechanics from Policy**

---

## Overview

The V2 architecture introduces a clean separation between **mechanics** (what the system *can* do) and **policy** (what the system *should* do). This makes the framework production-ready and accessible to different personas.

---

## Design Principles

### 1. Separation of Concerns

**Before (V1):**
```python
# Mechanics and policy mixed together
learner = OckhamLearner(
    model=model,
    surprise_threshold=2.0,  # Policy
    lambda_ockham=0.01,      # Policy
)
# Hard to change policy without touching code
```

**After (V2):**
```python
# Mechanics: OccamContext + Plugins
ctx = OccamContext(iter_num=100, task_loss=1.5, ...)
plugins = [OckhamGatePlugin(), CompressorPlugin(), ...]

# Policy: Configuration
for plugin in plugins:
    ctx = plugin.on_batch_start(ctx)  # Plugins modify hyperparameters
```

**Benefits:**
- Policy can be changed via config files (YAML)
- Mechanics remain stable and testable
- Different personas can use different policies

---

### 2. Standardized Context Object

**OccamContext** is the "VST-Rack" pattern - a standardized data object that flows through the plugin chain.

```python
@dataclass
class OccamContext:
    # Immutable core data
    iter_num: int
    task_loss: float
    
    # Mutable hyperparameters (plugins can modify)
    learning_rate: float
    lambda_ockham: float
    surprise_threshold: float
    
    # Metrics
    complexity_cost: float
    grad_norm: float
    update_rate: float
    
    # Occam Quotient
    model_params: int
    memory_footprint_mb: float
    inference_flops: float
    
    def compute_occam_quotient(self, task_performance: float) -> float:
        """OQ = Performance / (Cost × Memory)"""
        ...
```

**Key Features:**
- Immutable core data (plugins read but don't modify)
- Mutable hyperparameters (plugins can adjust)
- Metrics tracking (plugins can add custom metrics)
- Occam Quotient calculation

---

### 3. Plugin Architecture

**OccamPlugin** is the base class for all plugins.

```python
class OccamPlugin(ABC):
    @abstractmethod
    def on_batch_start(self, ctx: OccamContext) -> OccamContext:
        """Modify hyperparameters before training step"""
        pass
    
    def on_batch_end(self, ctx: OccamContext) -> None:
        """React to metrics after training step"""
        pass
    
    def on_consolidate(self, ctx: OccamContext) -> None:
        """Called when anchor is updated"""
        pass
```

**Implemented Plugins:**
- **OckhamGatePlugin:** Surprise gate (skip updates when loss < threshold)
- **CompressorPlugin:** Audio compressor-inspired dynamic hyperparameter control
- **LimiterPlugin:** Hard caps on complexity/gradients

---

### 4. Intelligent Model Selection

**OckhamMemoryV2** decides whether to accept new model states based on Ockham's Razor.

```python
def should_accept_update(
    self,
    ctx: OccamContext,
    prev_snapshot: Optional[ModelSnapshot],
) -> Tuple[bool, str]:
    """
    Three-gate decision logic:
    1. Quality gate: loss < quality_threshold
    2. Complexity gate: complexity_cost < complexity_threshold
    3. Efficiency gate: OQ improvement > min_oq_improvement
    
    Returns: (accept: bool, reason: str)
    """
```

**Rejection Reasons:**
- `REJECTED_QUALITY`: Loss too high
- `REJECTED_COMPLEXITY`: Drift too large
- `REJECTED_OQ_INSUFFICIENT`: Not efficient enough

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   nanoGPT-Ockham V2                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              OccamContext (Data)                     │  │
│  │  - Immutable: iter_num, task_loss                    │  │
│  │  - Mutable: learning_rate, lambda_ockham             │  │
│  │  - Metrics: complexity_cost, grad_norm, OQ           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Plugin Chain (Mechanics)                   │  │
│  │  1. OckhamGatePlugin → Skip updates                  │  │
│  │  2. CompressorPlugin → Adjust λ/LR                   │  │
│  │  3. LimiterPlugin → Hard caps                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        OckhamMemoryV2 (Decision Logic)               │  │
│  │  - should_accept_update() → 3 gates                  │  │
│  │  - Occam Quotient comparison                         │  │
│  │  - Rejection reason logging                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      nanoGPT (Base)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Persona-Based Usage

### Infra/Platform Teams
**Goal:** Stability, observability, compliance

**What they care about:**
- Structured logging (JSONL with `reason_code`)
- Metrics tracking (update rate, accept rate, OQ)
- Reproducibility (config-driven)

**How they use it:**
```python
# Load production config
memory = OckhamMemoryV2(
    quality_threshold=1.5,
    complexity_threshold=0.1,
    min_oq_improvement=0.05,
)

# Log all decisions
accept, reason = memory.should_accept_update(ctx, prev)
logger.info({"decision": accept, "reason": reason, "oq": ctx.compute_occam_quotient(...)})
```

---

### Research/ML Engineers
**Goal:** Experimentation with guardrails

**What they care about:**
- Easy to swap plugins
- Custom metrics
- Ablation studies

**How they use it:**
```python
# Experiment: Remove compressor, add custom plugin
plugins = [
    OckhamGatePlugin(surprise_threshold=2.0),
    MyCustomPlugin(param=0.5),
    LimiterPlugin(complexity_ceiling=0.2),
]

# Run ablation
for plugin_combo in itertools.combinations(plugins, 2):
    run_experiment(plugin_combo)
```

---

### Product Teams
**Goal:** ROI metrics, simple presets

**What they care about:**
- Occam Quotient (efficiency metric)
- Presets (no hyperparameter tuning)
- Cost savings

**How they use it:**
```python
# Use preset
from presets import load_preset
plugins, memory = load_preset("production_balanced")

# Track ROI
oq_before = baseline_oq
oq_after = ctx.compute_occam_quotient(...)
roi_improvement = (oq_after - oq_before) / oq_before
print(f"Efficiency improved by {roi_improvement*100:.1f}%")
```

---

## Occam Quotient (OQ)

**Definition:**
```
OQ = Task Performance / (Inference Cost × Memory Footprint)
```

**Interpretation:**
- **Higher is better** - we want high performance with low cost
- Balances quality and efficiency
- Stakeholder-friendly metric

**Example:**

| Model | Loss | Params | Memory | FLOPs | Performance | OQ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| A (13B) | 1.2 | 13B | 26GB | 2.6T | 0.833 | 1.23e-17 |
| B (7B) | 1.3 | 7B | 14GB | 1.4T | 0.769 | **3.92e-17** |

**Result:** Model B wins! 218% better OQ despite slightly worse loss.

---

## Decision Logic

### Three-Gate System

```python
def should_accept_update(ctx, prev):
    # Gate 1: Quality
    if ctx.task_loss > quality_threshold:
        return False, "REJECTED_QUALITY"
    
    # Gate 2: Complexity
    if ctx.complexity_cost > complexity_threshold:
        return False, "REJECTED_COMPLEXITY"
    
    # Gate 3: Efficiency (OQ)
    current_oq = ctx.compute_occam_quotient(...)
    prev_oq = prev.metrics.occam_quotient
    improvement = (current_oq - prev_oq) / prev_oq
    
    if improvement < min_oq_improvement:
        return False, "REJECTED_OQ_INSUFFICIENT"
    
    return True, "ACCEPTED_OQ_IMPROVEMENT"
```

**Statistics Tracked:**
- Total evaluations
- Accepted updates
- Rejected (quality, complexity, OQ)
- Accept rate

---

## Key Metrics

### 1. Update Rate (Gate)
**Definition:** Fraction of batches that triggered updates

**Typical Values:**
- Tight: 20-40% (60-80% compute saved)
- Balanced: 50-70% (30-50% compute saved)
- Exploratory: 80-95% (5-20% compute saved)

### 2. Accept Rate (Memory)
**Definition:** Fraction of models accepted for saving

**Typical Values:**
- Tight: 1-5% (95-99% storage saved)
- Balanced: 5-15% (85-95% storage saved)
- Exploratory: 15-30% (70-85% storage saved)

### 3. Occam Quotient (OQ)
**Definition:** Efficiency metric (performance / cost)

**Usage:**
- Compare models of different sizes
- Track efficiency over training
- Justify model selection to stakeholders

---

## Migration from V1 to V2

### Step 1: Replace TrainingState with OccamContext

**Before:**
```python
state = TrainingState(iter_num=100, task_loss=1.5, ...)
```

**After:**
```python
ctx = OccamContext(iter_num=100, task_loss=1.5, ...)
```

### Step 2: Update Plugins

**Before:**
```python
class MyPlugin(LearningPlugin):
    def on_batch_start(self, state: TrainingState) -> TrainingState:
        ...
```

**After:**
```python
class MyPlugin(OccamPlugin):
    def on_batch_start(self, ctx: OccamContext) -> OccamContext:
        ...
```

### Step 3: Use OckhamMemoryV2

**Before:**
```python
memory = OckhamMemory()
memory.add_candidate(...)
best = memory.select_best(...)
```

**After:**
```python
memory = OckhamMemoryV2()
accept, reason = memory.should_accept_update(ctx, prev)
if accept:
    snapshot = memory.save_checkpoint(model, ctx)
```

---

## Testing

### Unit Tests

```bash
# Test individual components
python ockham_context.py      # OccamContext + OQ
python plugins_v2.py           # Plugins with OccamContext
python ockham_memory_v2.py     # Memory decision logic
```

### Integration Test

```bash
# Test complete system
python demo_core_refactoring.py
```

**Expected Output:**
- Update rate: 70-90%
- Accept rate: 2-10%
- OQ tracking working
- Rejection reasons logged

---

## Future Extensions

### 1. Structured Logging
```python
import json

def log_decision(ctx, accept, reason):
    log_entry = {
        'timestamp': time.time(),
        'iter_num': ctx.iter_num,
        'decision': 'ACCEPT' if accept else 'REJECT',
        'reason_code': reason,
        'metrics': ctx.to_dict(),
    }
    with open('decisions.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

### 2. KL-Divergence Watchdog
```python
class KLWatchdogPlugin(OccamPlugin):
    def on_batch_end(self, ctx):
        kl_div = compute_kl_divergence(current_output, anchor_output)
        if kl_div > self.threshold:
            raise ValueError("Model diverged too far from anchor!")
```

### 3. Occam Decision Records (ODR)
```markdown
# ODR-001: Enable TTT for User-Specific Adaptation

## Problem
Current model accuracy drops 15% on user-specific vocabulary.

## Proposed Change
Enable TTT on layers 18-24 with LoRA adapter.

## Occam Constraints
- [ ] Reversibility: Can revert in < 10ms
- [ ] Complexity Cap: Added params < 5% of base
- [ ] Quality Gate: Val loss improves by > 2%

## Outcome
OQ Delta: +0.4 → KEEP
```

---

## Summary

**V2 Architecture Benefits:**

| Aspect | V1 | V2 |
|:---|:---|:---|
| **Mechanics/Policy** | Mixed | Separated |
| **Context Object** | TrainingState | OccamContext |
| **Plugin Interface** | LearningPlugin | OccamPlugin |
| **Decision Logic** | Implicit | Explicit (3 gates) |
| **Efficiency Metric** | None | Occam Quotient |
| **Observability** | Limited | Full (rejection reasons) |
| **Persona-Friendly** | No | Yes |

**Key Takeaway:**
The V2 architecture makes nanoGPT-Ockham production-ready by cleanly separating what the system *can* do (mechanics) from what it *should* do (policy).

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-14  
**Related:** README.md, PLUGIN_SYSTEM.md, FUTURE_DIRECTIONS.md
