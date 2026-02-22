# movielens-matrix-completion

This project implements and compares three algorithms for low-rank matrix completion on the MovieLens dataset, developed as part of the *Dimension Reduction* lab course (Ludovic Stephan).

The core problem is the following: given a matrix of user-movie ratings where most entries are missing, can we reconstruct the full matrix by assuming it has low rank? This is essentially the problem Netflix faced during its famous Prize competition, and it turns out to be deeply connected to sparse linear regression — just replacing vector sparsity with matrix low-rankness.

---

## What's in the notebook

We start by exploring the MovieLens dataset and analyzing the singular value spectrum of the observed rating matrix, which motivates the low-rank assumption. We then implement three algorithms from scratch:

**SVP (Singular Value Projection)** takes a hard approach: at each iteration it projects the current estimate onto the set of rank-$r$ matrices via truncated SVD. This is the matrix completion analogue of Iterative Hard Thresholding in sparse regression.

**SVT (Singular Value Thresholding)** takes a softer, convex approach: instead of enforcing a hard rank constraint, it penalizes the nuclear norm (sum of singular values), which is the convex relaxation of the rank. At each iteration it applies soft-thresholding to the singular values. This mirrors LASSO in the sparse regression setting.

**ADMiRA (Atomic Decomposition for Minimum Rank Approximation)** takes a greedy approach: it builds the solution iteratively as a linear combination of rank-1 atoms, expanding and pruning a dictionary at each step — directly analogous to Orthogonal Matching Pursuit.

All algorithms are evaluated on a fixed train/validation/test split (60/20/20) using the relative error on observed entries as the primary metric, with RMSE reported for interpretability. Hyperparameters (rank $r$ for SVP and ADMiRA, threshold $\lambda$ for SVT) are selected via cross-validation on the validation set.

---

## Results in brief

ADMiRA with $r=3$ achieves the best validation error (0.4450) and is also the fastest (58s over 30 iterations). SVP with $r=4$ is a close second (val error 0.4535, 90s). SVT lags behind both in accuracy and speed (val error 0.5222, 158s), mainly because it requires a full SVD at every iteration and is slower to converge. Across all three algorithms, a very low effective rank suffices — consistent with the singular value analysis showing that predictive structure concentrates in few components.

---

## Requirements

```
numpy
scipy
pandas
matplotlib
seaborn
```

```bash
pip install numpy scipy pandas matplotlib seaborn
```

---

## Authors

Emidio Grillo & Luca Nudo
