# Neural Collaborative Recommender System

A custom two-tower neural network built entirely from scratch using **pure NumPy**. This system implements deep-learning-based collaborative filtering to match users with highly compatible properties in the Auckland housing market.

By avoiding high-level frameworks like TensorFlow or PyTorch, this project explicitly implements the underlying linear algebra, matrix shapes, forward/backward propagation, and gradient descent optimization routines.

---

## Project Architecture and File Structure

The project is structured modularly to isolate data engineering from the core algorithmic network brain:

```text
Neural_Collaborative_Recommender_System/
├── generate.py        # Data pipeline: Synthetic generation, scaling, & tensor alignment
├── model.py           # Core brain: Matrix initialization, activations, forward & backprop
├── main.py            # Execution wrapper: Training orchestration and validation
├── .gitignore         # Prevents tracking cache artifacts
└── README.md          # Project documentation
```

---

## Project Components

- **[View Data Generation Notebook (Alternative Link)](https://nbviewer.org/github/ShivaramHR/Neural_Collaborative_Recommender_System/blob/main/generate.ipynb)** - Step-by-step synthetic user and house feature matrix creation pipeline.
* **`main.py`** - Core training script utilizing the optimized model configuration.
  > **Metrics:** `Training Accuracy: 95.58%` | `Blind Test Accuracy: 90.74%`
- `model.py` - Custom Multi-Layer Neural Collaborative Filtering Two-Tower architecture built from scratch.

---

### 🚀 Optimization & Regularization (Phase 1)
Initially, the unregularized Two-Tower network overfitted the 10k dataset, yielding a wide generalization gap (96.16% Train vs. 88.72% Test). 

To resolve this, I implemented custom $L_2$ Regularization and Inverted Dropout from scratch:
* **The Math Bottleneck:** Discovered an implementation hurdle where applying the dropout mask to the post-activation gradient ($dA$) misaligned with the activation caches. Corrected the backpropagation chain rule to apply the mask to the pre-activation gradient ($dZ$), eliminating gradient explosion/vanishing issues.
* **Plateau Traversal:** Encountered a distinct optimization plateau at `88.02%` accuracy. Solved this by executing a multi-stage training strategy, utilizing a higher fixed learning rate (`0.075`) to traverse saddle points before dropping to a fine-tuning rate (`0.005`) to let the weights settle into the global minimum.
* **Final Phase 1 Metrics:** Hitting **92.03% Training Accuracy** and **88.58% Test Accuracy**successfully shrinking the generalization gap to a highly stable **3.45%**.

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~96% |
| Dataset | Synthetically generated Auckland housing interactions |

---

## What This Project Covers

This is my first deep learning project, built after completing Course 1 of the Deep Learning Specialization. The goal was to go beyond the theory and implement everything by hand, including:

- **Weight initialization** using He initialization to prevent vanishing/exploding gradients
- **Forward propagation** with manual matrix multiplications across all layers
- **Activation functions** (ReLU, Sigmoid) implemented from scratch
- **Cost computation** using binary cross-entropy
- **Backward propagation** deriving all gradients by hand with the chain rule
- **Gradient descent** parameter updates across both towers

---

## The Two-Tower Architecture

The model uses a two-tower design:

- **User Tower** takes user interaction features as input and learns a user embedding
- **Property Tower** takes property listing features as input and learns a property embedding
- Both towers output vectors that are compared via dot product similarity to produce a compatibility score

---

## Tech Stack

| Tool | Role |
|------|------|
| NumPy | All neural network math and matrix operations |
| Pandas | Data loading, preprocessing, and feature engineering |

No PyTorch. No TensorFlow. No Keras. Everything is hand-rolled.

---

## Context

This is a learning project. The dataset is synthetically generated to simulate Auckland housing market interactions. The focus is on the mechanics of neural networks, not production accuracy.

Built from scratch after completing **Course 1 (Neural Networks and Deep Learning)** of the Deep Learning Specialization by Andrew Ng.
