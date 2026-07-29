# SpectralNet — Implementation Guide
### Paper: "SpectralNet: Spectral Clustering Using Deep Neural Networks" (arXiv 1801.01587)

---

## Table of Contents

1. [Motivation — Why SpectralNet?](#1-motivation)
2. [Big Picture: What You Are Building](#2-big-picture)
3. [Component 1 — Siamese Network (Metric Learning)](#3-siamese-network)
4. [Component 2 — Gaussian Affinity / k-NN Graph](#4-affinity-graph)
5. [Component 3 — SpectralNet (The Core Network)](#5-spectralnet-core)
6. [Component 4 — Orthogonalization via Cholesky](#6-cholesky-orthogonalization)
7. [Component 5 — Spectral Loss Function](#7-spectral-loss)
8. [Component 6 — Final Clustering (k-Means)](#8-final-clustering)
9. [Data Pipeline](#9-data-pipeline)
10. [Training Recipe — Full End-to-End](#10-training-recipe)
11. [Final Structure Summary](#11-final-structure)

---

## 1. Motivation

### Why can't we just use classical Spectral Clustering?

Classical Spectral Clustering (Ng, Jordan & Weiss 2002) works beautifully in theory:
1. Build a similarity graph W over your data.
2. Compute the graph Laplacian L = D − W.
3. Find the k smallest eigenvectors of L.
4. Those eigenvectors form a low-dimensional "spectral embedding."
5. Run k-means on that embedding to get clusters.

**The problem:** Computing eigenvectors of L requires O(n³) time and O(n²) memory. For a dataset with n = 70,000 samples (like MNIST), this is completely infeasible.

**The deeper problem:** Even if you compute the eigenvectors, they are defined only for points in your training set. A new test point has no natural embedding — the model cannot generalize.

### What SpectralNet does differently

SpectralNet trains a **deep neural network** whose outputs, for a given mini-batch, approximate the eigenvectors of the graph Laplacian **of the full dataset**. Specifically:

- The network `f_θ(x)` maps raw inputs x ∈ ℝ^d → ℝ^k (k = number of clusters).
- The network is trained so its outputs satisfy the **same conditions** that eigenvectors satisfy — orthonormality and minimization of the Rayleigh quotient.
- Because it is a neural network, it **generalizes** to unseen points automatically.
- Mini-batch training makes it **scalable** to large datasets.

This is the core insight. Everything else is engineering to make this work.

---

## 2. Big Picture: What You Are Building

The full pipeline has these stages, which must be implemented in order:

```
Raw Data
   │
   ▼
[Optional] Siamese Network → learned metric / embedding space
   │
   ▼
Affinity Matrix W (built from k-NN graph, using Gaussian kernel)
   │
   ▼
SpectralNet (MLP) trained with Spectral Loss + Cholesky orthogonalization
   │
   ▼
Spectral Embedding Y ∈ ℝ^{n × k}
   │
   ▼
k-Means on Y → Cluster Assignments
```

The Siamese network is **optional** but strongly recommended for image data. It learns a metric so that the affinity graph is more meaningful than raw pixel distances.

**Running example throughout this guide:** MNIST (60,000 train + 10,000 test images, 28×28 = 784 dimensions, 10 classes/clusters).

---

## 3. Component 1 — Siamese Network (Metric Learning)

### Motivation

If you build a k-NN graph directly in pixel space (784 dimensions), the distances are noisy and high-dimensional (curse of dimensionality). A Siamese network learns a mapping `g_φ: ℝ^784 → ℝ^d_embed` such that similar images are close and dissimilar images are far in the embedding space. The affinity graph is then built in this learned space.

### When do you need it?

- **Image data (MNIST, CIFAR-10):** Yes, use Siamese.
- **Low-dimensional data (Reuters text TF-IDF, 2D toy data):** Often skip it and use raw features directly.

### What is a Siamese Network?

Two copies of the same network (shared weights) take two inputs (x_i, x_j) and produce embeddings. A contrastive loss is used:
- If x_i and x_j are in the **same class** (positive pair): push their embeddings closer.
- If they are in **different classes** (negative pair): push embeddings apart, but only if they are closer than a margin.

### How to get pairs for MNIST?

You need label information only for the **Siamese training**, not for SpectralNet itself. For MNIST, you have labels, so:
- **Positive pairs:** randomly sample two images with the same digit label.
- **Negative pairs:** randomly sample two images with different digit labels.

At training time, sample a batch of B pairs. Use B = 256 pairs (128 positive + 128 negative).

### Architecture of the Siamese sub-network (MNIST)

Each "arm" of the Siamese (the shared network g_φ) is a simple MLP:

| Layer | Input dim | Output dim | Activation |
|-------|-----------|------------|------------|
| FC 1  | 784       | 1024       | ReLU       |
| FC 2  | 1024      | 1024       | ReLU       |
| FC 3  | 1024      | 512        | ReLU       |
| FC 4  | 512       | d_embed    | Linear     |

**d_embed:** Set to 10 for MNIST (same as number of clusters). This is a hyperparameter — in the paper they use values like 10–512 depending on dataset.

The output is **not normalized** at this stage.

### Loss Function — Contrastive Loss

For a pair (x_i, x_j) with label y ∈ {0, 1} (0 = same class, 1 = different class):

```
dist = ||g_φ(x_i) - g_φ(x_j)||_2

loss_positive = (1 - y) * dist²
loss_negative = y * max(0, margin - dist)²

total_loss = mean(loss_positive + loss_negative) over batch
```

**margin:** Set to 2.0. This is the threshold below which negative pairs are penalized.

### Training the Siamese

- Optimizer: RMSprop (lr = 1e-3)
- Epochs: 10–15 epochs over the training set
- Batch size: 256 pairs
- No weight decay needed

After training, discard the loss function. Keep only g_φ. You will use it to **encode all training points** into d_embed-dimensional space before building the affinity graph.

---

## 4. Component 2 — Affinity Matrix / k-NN Graph

### Motivation

SpectralNet needs a notion of "similarity" between data points to define what "good clustering" means. This is encoded in an affinity matrix W, where W_ij is the similarity between points i and j.

You do **not** build a full n×n matrix. Instead you build it on mini-batches.

### Step-by-step: Building W for a mini-batch

Given a mini-batch of m points {x_1, ..., x_m} (already encoded by Siamese if applicable):

**Step 1: Compute pairwise distances**

Compute the m×m matrix of squared Euclidean distances between all pairs in the batch.

**Step 2: k-NN masking**

For each point x_i, find its k nearest neighbors within the batch. Set all non-neighbor entries to 0. This creates a sparse affinity. 

**k:** Set k = 10 for MNIST.

**Step 3: Gaussian kernel**

Convert distances to similarities using a Gaussian kernel:

```
W_ij = exp(- dist(x_i, x_j)² / (2 * σ²))
```

**σ (scale parameter):** This is crucial. The paper uses a **data-driven σ**: for each point, σ_i is the mean distance to its k nearest neighbors. Then the final kernel uses σ_i * σ_j (the geometric mean). This is called the "self-tuning" kernel from Zelnik-Manor & Perona.

So in practice:
```
σ_i = mean distance from x_i to its k nearest neighbors
W_ij = exp(- dist(x_i, x_j)² / (σ_i * σ_j))  [only for k-NN pairs, else 0]
```

**Step 4: Symmetrize**

W = (W + W^T) / 2

This gives you the mini-batch affinity matrix W ∈ ℝ^{m×m}.

---

## 5. Component 3 — SpectralNet (The Core Network)

### Architecture (MNIST)

SpectralNet is an MLP: f_θ: ℝ^784 → ℝ^k, where k = 10 (number of clusters).

| Layer | Input dim | Output dim | Activation |
|-------|-----------|------------|------------|
| FC 1  | 784       | 1024       | ReLU       |
| FC 2  | 1024      | 1024       | ReLU       |
| FC 3  | 1024      | 512        | ReLU       |
| FC 4  | 512       | k (10)     | Linear     |

**Important:** The final layer has **no activation**. Raw linear output. The orthogonalization is handled externally by Cholesky (next section).

**Note on the Siamese connection:** If you trained a Siamese network, SpectralNet takes **raw pixels** as input (not Siamese embeddings). The Siamese is used only to build W. SpectralNet learns its own mapping from raw space, guided by W.

For datasets where you skip Siamese (e.g., Reuters), the input is TF-IDF vectors, and the architecture shrinks accordingly (fewer/smaller layers).

---

## 6. Component 4 — Orthogonalization via Cholesky

### Motivation

Spectral clustering's eigenvectors are orthonormal: Y^T Y = I. We need SpectralNet's outputs to satisfy the same constraint. But we cannot use a hard constraint during gradient descent easily.

The solution: after every forward pass on a mini-batch, apply **Cholesky-based orthogonalization** to the output matrix. This is differentiable, so gradients flow through it.

### What is happening geometrically?

Let Y ∈ ℝ^{m×k} be the raw output of SpectralNet for a mini-batch of m points.

We want to find a linear transformation M such that (Y · M)^T (Y · M) = I (orthonormal columns).

This means: M^T (Y^T Y) M = I, i.e., M is the "whitening" matrix of Y.

### Cholesky Orthogonalization — Step by Step

**Step 1:** Compute the Gram matrix:
```
S = Y^T Y       ← shape: k × k
```

S is symmetric positive (semi)definite.

**Step 2:** Cholesky decompose S:
```
S = L L^T       ← L is lower-triangular, shape: k × k
```

**Step 3:** Compute the orthogonalized output:
```
Y_orth = Y · (L^T)^{-1}   ← shape: m × k
```

**Verification:** Y_orth^T Y_orth = (L^{-1}) Y^T Y (L^{-T}) = (L^{-1}) L L^T (L^{-T}) = I ✓

### Implementation note

Use `torch.linalg.cholesky(S)` to get L. Then use `torch.linalg.solve_triangular(L.T, Y.T, upper=True).T` to compute Y · (L^T)^{-1} without explicitly inverting L (numerically more stable).

Add a small regularizer for numerical stability:
```
S = Y^T Y + ε * I     (ε = 1e-4 or 1e-6)
```

This is done **inside the forward pass of SpectralNet**, so gradients flow through the Cholesky operation. PyTorch autograd handles this automatically since `torch.linalg.cholesky` is differentiable.

### Frequency of orthogonalization

Orthogonalize **every forward pass** on every mini-batch. There is no "separate orthogonalization step."

---

## 7. Component 5 — Spectral Loss Function

### Motivation

The eigenvectors of the graph Laplacian minimize the Rayleigh quotient:

```
Y* = argmin_Y  trace(Y^T L Y)  subject to  Y^T Y = I
```

where L = D − W is the unnormalized graph Laplacian, D is the degree matrix (D_ii = sum of row i of W).

SpectralNet minimizes exactly this objective on mini-batches, after Cholesky orthogonalization enforces the constraint.

### Computing the Spectral Loss on a Mini-Batch

Given:
- Y_orth ∈ ℝ^{m×k}: orthogonalized output of SpectralNet for the batch
- W ∈ ℝ^{m×m}: affinity matrix for the batch (computed in Component 2)

**Step 1:** Compute degree matrix D (diagonal):
```
d_i = sum_j W_ij     ← sum of row i of W
D = diag(d)
```

**Step 2:** Compute unnormalized Laplacian:
```
L = D - W
```

**Step 3:** Compute spectral loss:
```
loss = trace(Y_orth^T L Y_orth) / m
```

Expanding:
```
loss = trace(Y_orth^T D Y_orth) - trace(Y_orth^T W Y_orth)
```

Or equivalently (and more efficient to compute):
```
loss = (1/m) * sum_{i,j} W_ij * ||y_i - y_j||²
```

where y_i is the i-th row of Y_orth. This is the most intuitive form: **pull together connected points** (minimize distance weighted by affinity).

**The division by m** is important for stable loss magnitudes across different batch sizes.

### What the loss is doing intuitively

If W_ij is large (points i and j are similar), then the loss penalizes ||y_i - y_j||² being large. So similar points are pushed to have similar embeddings. The orthogonality constraint (via Cholesky) prevents the trivial solution of mapping everything to zero.

---

## 8. Component 6 — Final Clustering (k-Means)

After SpectralNet is trained:

1. **Encode all training points:** Run the full training set through f_θ to get Y ∈ ℝ^{n×k} (with Cholesky applied on a large batch or via the online approximation — see below).

2. **Normalize rows** (optional but recommended): Normalize each row of Y to unit length before k-means. This follows the standard Ng-Jordan-Weiss spectral clustering protocol.

3. **Run k-Means on Y:** Use sklearn's KMeans with k=10 clusters, n_init=10 (run 10 times from different random initializations, keep the best). 

4. **Evaluate:** Use Unsupervised Clustering Accuracy (ACC) — the best permutation match between cluster assignments and true labels via the Hungarian algorithm.

### Encoding the full dataset for final clustering

At inference time, there is no mini-batch constraint. Run the full dataset through f_θ in chunks (e.g., 1000 at a time) and apply Cholesky on the whole output matrix. The Cholesky at inference uses the full Y (not per-mini-batch), so the orthogonalization is global.

---

## 9. Data Pipeline

### MNIST Preprocessing

**Load:** Use torchvision or raw numpy from the original binary files.

**Normalize:** Scale pixels to [0, 1] by dividing by 255. Then standardize:
```
x = (x - mean) / std
```
where mean and std are computed over the training set (per-pixel or global — global is fine for MNIST).

**Flatten:** 28×28 → 784 vector. Do this before any network processing.

**No augmentation** is used in the SpectralNet paper.

### Data flow during SpectralNet training

At each training step:

1. Sample a random mini-batch of m=1024 raw images (indexes into training set).
2. Encode them with g_φ (Siamese network, frozen) → get embeddings in ℝ^{d_embed}.
3. Build affinity matrix W ∈ ℝ^{1024×1024} from these embeddings using k-NN + Gaussian kernel.
4. Pass the same m=1024 raw images through f_θ (SpectralNet) → get Y ∈ ℝ^{1024×k}.
5. Apply Cholesky to Y → get Y_orth.
6. Compute spectral loss using W and Y_orth.
7. Backpropagate only into f_θ (Siamese is frozen).

### Mini-batch size

**m = 1024** for MNIST. The paper notes that larger mini-batches give better approximation of the global Laplacian. Do not go below m = 256.

---

## 10. Training Recipe — Full End-to-End

### Phase 1: Train the Siamese Network

| Parameter       | Value         |
|-----------------|---------------|
| Optimizer       | RMSprop       |
| Learning rate   | 1e-3          |
| Batch size      | 256 pairs     |
| Epochs          | 10            |
| Margin          | 2.0           |
| Architecture    | 784→1024→1024→512→10 |

- Shuffle pairs each epoch.
- No learning rate schedule needed.
- Save the trained weights.
- After training: encode all training points into ℝ^10. This encoded dataset is used to build W (never passed to SpectralNet as input).

### Phase 2: Train SpectralNet

| Parameter       | Value         |
|-----------------|---------------|
| Optimizer       | RMSprop       |
| Learning rate   | 1e-3          |
| Batch size      | 1024 images   |
| Epochs          | 50            |
| k (in k-NN)     | 10            |
| ε (Cholesky regularizer) | 1e-4 |
| Architecture    | 784→1024→1024→512→10 |

**Learning rate schedule:** Halve the learning rate every 20 epochs. Alternatively, use a step scheduler: decay by factor 0.5 at epoch 20 and epoch 40.

**Training loop pseudocode (conceptual):**
```
for each epoch:
    shuffle training indices
    for each mini-batch of 1024 indices:
        x_batch = raw_images[indices]          # m × 784
        z_batch = siamese(x_batch)             # m × 10 (frozen)
        
        W = build_affinity(z_batch, k=10)     # m × m
        
        Y_raw = spectralnet(x_batch)           # m × k (linear output)
        Y_orth = cholesky_orthogonalize(Y_raw) # m × k
        
        loss = spectral_loss(Y_orth, W)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**What to monitor during training:**
- Spectral loss should decrease from ~0.5–1.0 and stabilize around ~0.05–0.2 for MNIST.
- If loss goes to 0 immediately and stays there — something is wrong (likely Cholesky is collapsing, check ε).
- If loss doesn't decrease at all after 10 epochs — check that gradients are actually flowing (inspect requires_grad).

### Phase 3: Final k-Means

| Parameter       | Value         |
|-----------------|---------------|
| k               | 10 (= n_clusters) |
| n_init          | 10            |
| max_iter        | 300           |

Run on Y_full (all training points encoded by SpectralNet + global Cholesky), with row normalization.

Expected ACC on MNIST: ~0.80–0.92 depending on implementation quality.

---

## 11. Final Structure Summary

```
spectralnet/
├── siamese.py          ← Siamese MLP + contrastive loss + training loop
├── affinity.py         ← k-NN graph + self-tuning Gaussian kernel
├── spectralnet.py      ← SpectralNet MLP + Cholesky orthogonalization
├── loss.py             ← Spectral loss (trace of Rayleigh quotient)
├── train.py            ← End-to-end training: Siamese → SpectralNet → k-Means
├── data.py             ← MNIST loader, normalization, pair sampling
├── cluster.py          ← k-Means + ACC evaluation with Hungarian algorithm
└── README.md
```

### Data flow (one sentence per component)

1. **data.py:** Load MNIST, normalize to zero mean unit variance, produce flat 784-dim vectors, sample positive/negative pairs for Siamese.
2. **siamese.py:** Train MLP [784→1024→1024→512→10] with contrastive loss; encode all train points.
3. **affinity.py:** For each mini-batch, compute pairwise distances in Siamese space, find k=10 NN per point, apply self-tuning Gaussian kernel, symmetrize → W.
4. **spectralnet.py:** MLP [784→1024→1024→512→10], forward pass returns raw output; Cholesky applied inside forward to produce Y_orth.
5. **loss.py:** Compute (1/m) * trace(Y_orth^T L Y_orth) given Y_orth and W.
6. **train.py:** Outer loops for both training phases; saves checkpoints after Siamese phase.
7. **cluster.py:** Run k-Means on global spectral embedding; evaluate ACC via Hungarian matching.

### Critical implementation traps to avoid

- **Do not pass Siamese embeddings into SpectralNet.** Siamese embeddings are only used for building W. SpectralNet always takes raw pixels as input.
- **Do not freeze the Cholesky.** It must be inside the computational graph so gradients flow through it.
- **Do not build W from raw pixels.** Always use the Siamese embedding space for the affinity graph (on image data).
- **Use self-tuning σ** (per-point, from NN distances), not a global fixed σ. Global σ is very sensitive to scale and hard to tune.
- **Do not normalize Y before Cholesky.** Cholesky handles normalization. If you normalize before, the Gram matrix S = Y^T Y ≈ I already, and Cholesky becomes a no-op.
- **Mini-batch affinity only.** Never try to compute the full n×n W matrix. SpectralNet's power is precisely that it approximates the global solution from mini-batch samples.