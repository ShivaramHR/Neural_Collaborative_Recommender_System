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
  > **Metrics:** `Training Accuracy: 95.48%` | `Blind Test Accuracy: 88.88%`
- `model.py` - Custom Multi-Layer Neural Collaborative Filtering Two-Tower architecture built from scratch.

---

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
