# Neural network-based point process models

Synthetic point processes are simulated using [`tick`](https://github.com/X-DataInitiative/tick)
and [`point-process-rust`](https://github.com/ManifoldFR/point-process-rust).

Neural networks are written using [PyTorch](https://pytorch.org).

## Objective

Use a modified neural network-based Hawkes process model for next event prediction in a time series.

## Overview

**Directories**:

* `data` contains the data
* `biblio` contains the bibliography
* `rapport` contains the project report
* `notebooks` contains the Jupyter notebooks

## Installation

### Loading a model

From [Saving & Loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).  
If the model state dict was saved with `torch.save(model.state_dict(), PATH)`


```python
import torch
from models import ModelClass

model = ModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval() # Evaluation mode
```

## References

1. Second order statistics characterization of Hawkes processes and non-parametric estimation (arXiv :1401.0903) E. Bacry, J.F. Muzy. Trans. in Inf. Theory, 62, Iss.4 (2016) https://arxiv.org/abs/1401.0903
2. Estimation of slowly decreasing Hawkes kernels : Application to high frequency order book modelling (arXiv :1412.7096) E.Bacry, T.Jaisson, J.-F.Muzy Quantitative Finance Vol.16 Iss. 8 (2016)  <https://arxiv.org/abs/1412.7096>
3. Hawkes processes in finance. (arXiv :1502.04592) E.Bacry, I.Mastromatteo, J.-F.Muzy Market Microstructure and Liquidity Vol. 01, No. 01, 1550005 (2015).  https://arxiv.org/abs/1502.04592
4. The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process, H. Mei, J. Eisner  <https://arxiv.org/pdf/1612.09328.pdf>
5. Modeling The Intensity Function Of Point Process Via Recurrent Neural Networks <https://arxiv.org/pdf/1705.08982.pdf>
6. On a Bayesian RNN for learning the decrease speed parameter in a process: Neural Hawkes Process Memory (Mike Mozer)  <http://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/talks/Mozer_NeuralHawkesProcessMemory_NIPS2016.pdf>
7. Recurrent Marked Temporal Point Processes:
<https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf>
8. Numerical Recipes The Art of scientific Computing (An amazing book for not analytic integration)
https://e-maxx.ru/bookz/files/numerical_recipes.pdf