# N:M Pruning

## Overview
This is an implementation of the method proposed in this [paper](https://openreview.net/forum?id=K9bw7vqp_s).
A model is trained from scratch together with a pruning mask for each layer, following the N:M semi-structured pattern.
At training time, sparsity is only simulated during the forward pass, while the backward remains dense.
The original source code can be found [here](https://github.com/aojunzz/NM-sparsity).

## Basic Usage
To convert a DNN only two lines of code are necessary:
```python
from plinio.methods import NMPruning
nm_model = NMPruning(model, n=1, m=4, input_shape=model_input_shape)
```
then a standard pytorch loop can be used to train the model's weights and sparsity masks.
Finally, the sparse model can be extracted with:
```
sparse_model = nm_model.export()
```

## Limitations
1. Only fully-connected and conv2d layers are currently supported.
2. N:M pruning is applied on weight tensors in the format KHWC, making both the forward and the backward passes slow on larger networks during the training.



