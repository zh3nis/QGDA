# EDA
Epanechnikov Discriminant Analysis

`qgda.py` provides one `SimplexQGaussianHead` for the compact-support
family `q < 1`. TP (Epanechnikov) is `q=0`; biweight is `q=0.5`.
Class means are fixed regular-simplex vertices with pairwise distance `s=6`
by default. Only the class priors are trainable.

```python
from qgda import SimplexQGaussianHead, qgaussian_joint_fy_loss, QGaussianRegFYLoss

head = SimplexQGaussianHead(C=10, D=9, q=0.5)
scores = head(embeddings)  # joint densities, shape (batch, 10)
loss = qgaussian_joint_fy_loss(head, embeddings, labels).mean()

# Optional class-wise projected-CF matching: use the same q as the head.
criterion = QGaussianRegFYLoss(D=9, q=0.5, q_weight=1.0)
regularized_loss = criterion(head, embeddings, labels, step=0)
```

With `k=1/(1-q)`, the class density is `[(-tau-Q/2)/k]_+^k`, where
`Q=||z-mu_c||^2/variance`. The fixed spherical matrix `Sigma=variance*I`
parameterizes the quadratic form. Its default gives actual moment covariance
`I/D` and support radius squared `(D+2*k+2)/D`. An explicit `variance`
overrides the shape scale. Normalization is computed in log space.

Scores are densities, and points outside every class support have all-zero
scores. The joint Fenchel–Young loss uses `alpha=2-q` and retains gradients
outside support. See [Martins et al., Section 6](https://jmlr.org/papers/v23/21-0879.html)
for the compact q-Gaussian family and Fenchel–Young framework.

Use `q=0` (the head's default) for TP and `q=0.5` for biweight.
When loading a state dictionary, construct the head with the same q used for
training.

`DeepQGaussian.ipynb` follows `DeepTPDA.ipynb` with the same three TP input
classes and two-hidden-layer backbone, using the fixed simplex head and
`QGaussianRegFYLoss` from `qgda.py`. Set `q < 1` (default `0.5`) and
`q_weight` (default `1.0`) in its setup cell.
It reports accuracy and coverage and plots learned embeddings, support
boundaries, decision regions, and training curves.

`Synthetic_TPReg.ipynb` and `Synthetic_QGaussianReg.ipynb` compare joint FY
loss against FY plus class-wise CF matching at `q=0` and `q=0.5`, respectively,
under matched initialization and batch order. They include support-aware
predictions, training history, and embedding snapshots. Both use the shared
head with an explicit q. Their density plots remain specialized to their
stated q values.

`ClasswiseQGaussianReg` and `QGaussianRegFYLoss` in `qgda.py` accept
`q < 1` (default `0.5`); pass `q=0` for TP. Residuals use fixed class centers
and the head's actual moment variance; the Gaussian window is a positive
integration weight. The combined loss rejects mismatched head and regularizer
q values. `sample_qgaussian_residuals` also accepts `q < 1` (default `0.5`),
including `q=0` for exact TP samples. All head, loss, regularizer, and sampler
implementations live in `qgda.py`.

Run the complete numerical suite with
`python -m unittest test_simplex_tpda test_tpreg test_qgaussian test_general_q`.
