# Synthetic data

Synthetic data following the Hawkes process distribution.
Must be processed into a PyTorch tensor upon loading.

**Type** : Python `dict`

**Layout**:
* `mu`: baseline (background) process intensity
* `alpha`: jump size (adjacency in the multivariate case)
* `decay`: event decay rate
* `timestamps`: event timestamps
* `types`: event types
* `lengths`: event sequence lengths