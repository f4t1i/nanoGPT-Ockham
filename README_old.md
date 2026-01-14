# Resilient Nano-Trainer: Adaptive Resonance Suppression

> **📦 STATUS: ARCHIVED (v1.0)** | This repository contains the complete ARS implementation and is maintained for reference. For the next phase (Tool-Selection Framework with ARS integration), see [nanoGPT-Agent-Framework](https://github.com/f4t1i/nanoGPT-Agent-Framework).

This repository demonstrates **Adaptive Resonance Suppression (ARS)**, a novel optimizer wrapper designed to enhance training stability in neural networks, particularly after distribution shifts. This implementation extends Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) to showcase ARS's effectiveness in a real-world, budget-constrained training scenario.

## 🚀 Key Results

Our experiments show that ARS significantly improves training robustness. When faced with an extreme data-shift (reversing the text dataset mid-training), the ARS-powered optimizer **survived 2x longer** than the standard AdamW optimizer and **completed the training without divergence**.

![Comprehensive Comparison](results/comprehensive_comparison.png)

| System | Divergence Step | Survival After Shift | Final Loss | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (AdamW)** | 650 | 350 steps | 2.099 | 🔴 **Diverged** |
| **ARS (Optimized)** | >1000 | **700+ steps** | **1.935** | 🟢 **Stable** |

## 💡 What is Adaptive Resonance Suppression?

ARS is a lightweight, plug-and-play wrapper for any PyTorch optimizer. It prevents training instability by using a three-layer defense mechanism:

1.  **Entropy Guard (Ψ_t)**: Detects periodicity and resonance in training dynamics using lag-1 autocorrelation of the loss "surprise" signal.
2.  **Surprise Gate (Φ_t)**: Adaptively damps gradient updates based on the magnitude of the surprise, preventing large, destabilizing steps.
3.  **Chronos-Jitter (χ_t)**: Injects a small amount of noise into gradients when resonance is detected, helping the model escape periodic attractors.

![ARS Mechanism](results/ars_mechanism.png)

This combination allows ARS to act as an intelligent circuit breaker, gently intervening only when it detects instability, without hindering normal training progress.

## 📚 Related Work

**Phase 2 (Upcoming):** [nanoGPT-Agent-Framework](https://github.com/f4t1i/nanoGPT-Agent-Framework)
- Tool-Selection Training System using ARS
- 4 Training Methods: SFT, RL, ICL, Continuous Learning
- 373 Tools Inventory
- ARS-Stabilized Reward System

## 🔬 Experiments & Replication

This repository contains all the code and data needed to replicate our findings.

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Matplotlib

### Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/f4t1i/Resilient-Nano-Trainer.git
    cd Resilient-Nano-Trainer
    ```

2.  **Prepare the dataset:**
    ```bash
    python data/shakespeare_char/prepare.py
    ```

3.  **Run the experiments:**

    *   **Optimized ARS (should succeed):**
        ```bash
        python ars_tuned_experiment.py
        ```

    *   **Baseline (should diverge):**
        ```bash
        python baseline_experiment.py
        ```

4.  **Generate plots:**
    ```bash
    python generate_plots.py
    ```

### How to Use ARS in Your Own Project

Using ARS is simple. Just wrap your existing optimizer:

```python
from ars_optimizer import ARSOptimizer
import torch.optim as optim

# 1. Define your base optimizer
base_optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 2. Wrap it with ARS
optimizer = ARSOptimizer(base_optimizer, alpha=1.0, phi_min=0.3)

# 3. In your training loop, pass the loss to optimizer.step()
for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step(loss.item()) # Pass loss value here
    optimizer.zero_grad()
```

## 📄 Scientific Documentation

For a detailed explanation of the theory, mathematics, and implementation, please see the [**MEMO.md**](MEMO.md) file.

## 📈 Conclusion

Adaptive Resonance Suppression is a powerful and easy-to-use technique for improving the stability and robustness of neural network training. By intelligently detecting and mitigating instability, ARS can help save significant compute resources in budget-constrained scenarios and enable more aggressive hyperparameter tuning.

This work demonstrates that with the right stability mechanisms, even small models can be trained to be resilient against extreme distribution shifts.

---

## Original nanoGPT README

![nanoGPT](assets/nanogpt.jpg)

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

(The rest of the original nanoGPT README follows...)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
