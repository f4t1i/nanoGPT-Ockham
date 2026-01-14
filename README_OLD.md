# nanoGPT-Ockham

> **Intelligent Neural Network Training with Ockham's Razor**

A clean, principled implementation of Ockham's Razor for neural network training and adaptation. Built on top of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), this framework operationalizes the philosophical principle of parsimony into measurable, actionable code.

## 🎯 Core Philosophy

**Ockham's Razor:** *"Entities should not be multiplied beyond necessity."*

In the context of neural networks, this translates to:
- **Training:** Only update parameters when the information gain justifies the complexity cost
- **Architecture:** Choose the simplest model that solves the problem adequately
- **Adaptation:** Make minimal changes to preserve existing knowledge

This framework makes these principles concrete, measurable, and automatic.

---

## 🚀 What Makes This Different?

Most training frameworks optimize for **maximum performance**. This framework optimizes for **minimal sufficient complexity**.

| Traditional Approach | nanoGPT-Ockham Approach |
|:---|:---|
| "Train on every batch" | "Only update when surprised" (Surprise Gate) |
| "Bigger is better" | "Simplest that works is best" (OckhamMemory) |
| "Fine-tune aggressively" | "Adapt minimally" (Anchor Regularization) |
| "Hope it doesn't forget" | "Actively protect knowledge" (Complexity Penalty) |

---

## 📦 Components

### 1. **OckhamLearner** - Intelligent Test-Time Training

A wrapper around any PyTorch model that implements controlled, minimal adaptation.

**Key Features:**
- **Surprise Gate:** Only updates weights when `task_loss > surprise_threshold`
- **Anchor Regularization:** Penalizes drift from a stable reference point
- **Complexity Tracking:** Measures and reports the "cost" of each adaptation

**Loss Function:**
```
L_total = L_task + λ_ockham * Ω(Δθ)
```

where `Ω(Δθ) = Σ ||θ - θ_anchor||²`

**Example Usage:**
```python
from ockham_learner import OckhamLearner

# Wrap your model
learner = OckhamLearner(
    model=model,
    optimizer=optimizer,
    lambda_ockham=0.01,           # How strongly to resist change
    surprise_threshold=0.5,        # Minimum loss to trigger update
    device='cuda'
)

# Adapt on new data
metrics = learner.adapt(inputs, targets, loss_fn)

# Consolidate when stable
learner.consolidate()
```

### 2. **OckhamMemory** - Intelligent Model Selection

Maintains a Pareto frontier of trained models and selects the simplest one that meets performance criteria.

**Key Features:**
- **Pareto Frontier:** Tracks models where no other is both simpler AND better
- **Automatic Selection:** Returns the simplest model meeting constraints
- **Persistent Storage:** Saves model metadata and checkpoints

**Example Usage:**
```python
from ockham_memory import OckhamMemory

# Initialize memory
memory = OckhamMemory(memory_dir='ockham_memory')

# Add candidates during training
memory.add_candidate(
    model=model,
    val_loss=1.234,
    train_loss=1.123,
    config=model_config
)

# Get the best (simplest sufficient) model
best = memory.get_best_model(max_val_loss=1.5)

# Load it
memory.load_best_model(model, max_val_loss=1.5)
```

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/f4t1i/nanoGPT-Ockham.git
cd nanoGPT-Ockham
```

2. **Prepare a dataset:**
```bash
python data/shakespeare_char/prepare.py
```

3. **Run the demo:**
```bash
python demo_ockham.py
```

This will show you how the OckhamLearner works on a simple toy problem.

---

## 📚 Training Examples

### Standard Training with Ockham Regularization

Train a model with anchor-based regularization (no surprise gate):

```bash
python train_ockham.py \
    --dataset=shakespeare_char \
    --use_ockham=True \
    --lambda_ockham=0.01 \
    --surprise_threshold=None
```

### Training with Surprise Gate

Only update when the loss exceeds a threshold:

```bash
python train_ockham.py \
    --dataset=shakespeare_char \
    --use_ockham=True \
    --lambda_ockham=0.01 \
    --surprise_threshold=2.0
```

This will skip updates on "easy" batches, saving computation and preventing overfitting to noise.

### Training with OckhamMemory

Enable model selection to find the simplest sufficient architecture:

```bash
python train_ockham.py \
    --dataset=shakespeare_char \
    --use_ockham=True \
    --use_ockham_memory=True \
    --lambda_ockham=0.01
```

After training, check the Pareto frontier:

```python
from ockham_memory import OckhamMemory

memory = OckhamMemory('out-ockham/ockham_memory')
print(memory.get_frontier_summary())
```

---

## 🔬 Understanding the Metrics

When using OckhamLearner, you'll see these metrics:

| Metric | Meaning |
|:---|:---|
| `task_loss` | The raw loss on the current batch (the "surprise") |
| `complexity_cost` | How far parameters have drifted from the anchor |
| `total_loss` | `task_loss + λ * complexity_cost` |
| `updated` | Whether weights were actually updated (False if gated) |
| `update_rate` | Fraction of batches that triggered updates |

**Interpretation:**
- **High `update_rate` (>80%):** Model is constantly surprised → increase `surprise_threshold` or decrease learning rate
- **Low `update_rate` (<20%):** Model is stable → consider consolidating or increasing learning rate
- **Rising `complexity_cost`:** Model is drifting from anchor → consider consolidating or increasing `lambda_ockham`

---

## 🧪 Experimental Results

*(To be added after running experiments)*

Key questions this framework helps answer:
1. **How much training is actually necessary?** (via surprise gate statistics)
2. **What's the simplest model that works?** (via OckhamMemory frontier)
3. **How much does fine-tuning cost in terms of knowledge drift?** (via complexity cost)

---

## 🎓 Theoretical Background

This implementation is grounded in several established principles:

1. **Minimum Description Length (MDL):** The best model is the one that compresses the data most effectively
2. **Elastic Weight Consolidation (EWC):** Protect important weights from change during continual learning
3. **Information Theory:** Only learn when the information gain exceeds the cost
4. **Pareto Efficiency:** Among models with similar performance, prefer the simpler one

The novelty is not in inventing new theory, but in creating a **unified, practical framework** that makes these principles operational and measurable.

---

## 🤝 Relationship to nanoGPT

This project is built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. We preserve the core simplicity and readability of nanoGPT while adding:
- `ockham_learner.py` (~250 lines) - The core adaptation logic
- `ockham_memory.py` (~300 lines) - Model selection logic
- `train_ockham.py` - Modified training script with Ockham integration

The original `model.py`, `train.py`, and data preparation scripts remain largely unchanged.

---

## 📖 Use Cases

This framework is particularly valuable for:

1. **Test-Time Training (TTT):** Adapt a pretrained model to new data without catastrophic forgetting
2. **Continual Learning:** Learn new tasks while preserving performance on old ones
3. **Resource-Constrained Deployment:** Find the smallest model that meets requirements
4. **Interpretable Training:** Understand *why* and *when* a model updates its weights
5. **Hyperparameter Tuning:** Use complexity cost as a regularization signal

---

## 🔮 Future Directions

Potential extensions:
- Integration with Adaptive Resonance Suppression (ARS) for extreme stability
- Multi-task OckhamMemory (Pareto frontier across multiple objectives)
- Automatic `lambda_ockham` tuning based on validation metrics
- Visualization tools for complexity cost and update patterns

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Andrej Karpathy** for [nanoGPT](https://github.com/karpathy/nanoGPT), the foundation of this work
- **William of Ockham** (1287-1347) for the philosophical principle that inspired this framework

---

## 📧 Contact

For questions, suggestions, or collaboration:
- GitHub Issues: [github.com/f4t1i/nanoGPT-Ockham/issues](https://github.com/f4t1i/nanoGPT-Ockham/issues)

---

**"Entities should not be multiplied beyond necessity."**  
*Let's build AI that respects this wisdom.*
