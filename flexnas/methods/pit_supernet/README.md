# PIT-SuperNet

## Overview
`plinio.pit_supernet` is a novel DNAS tool combining the lightweight and fine-grain search operated by [PIT](../pit/README.md) with the more flexible supernet-based approach inspired by [DARTS](https://arxiv.org/abs/1806.09055).

In general, a supernet-based DNAS allows to select part of networks from a pool of options. This gives an additional degree of flexibility to the user which will be able to explore alternatives such as the usage of *simple convolution versus depthwise-separable convolution versus no convolution at all (i.e., identity operation)*.

On top of that, `plinio.pit_supernet` is capable to explore in a *pit-like fashion* the [hyper-parameters](../pit/README.md#supported-layers) of the different specified layers choices.

## Basic Usage
`plinio.pit_supernet` exposes an API that shares many similarities with the one of [PIT](../pit/README.md#basic-usage). The main difference is found in the fact that the user should define a network with the different layer alternatives that desires to explore. Then, unless otherwise stated all the [PIT supported layers](../pit/README.md#supported-layers) will be automatically considered for optimization.

#### Search-Space Limitations
At the current stage, `plinio.pit_supernet` requires to define a pool of layer alternatives sharing **the same output tensor dimension**.

### Example of SuperNet definition and usage
The API provided by `plinio` to define the collection of layers among the tool should select an alternative is `PITSuperNetModule`.

This module simply requires a list of layer satisfying [PIT-SuperNet Requirements](#search-space-limitations).

E.g., we can define a simple supernet as:
```python
from torch import nn
from plinio.methods.pit_supernet import PITSuperNetModule

class SimpleSuperNet(nn.Module):
    def __init__(self):
        self.conv = PITSuperNetModule([
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding='same'),
                nn.Conv2d(32, 32, 3, padding='same'),
            ),
            nn.Conv2d(32, 32, 5, padding='same'),
            nn.Identity()
        ])

    def forward(self, x):
        return self.conv(x)
```
In this example, we are saying that we want that `plinio.pit_supernet` explores and select the list of different layer alternatives passed to `PITSuperNetModule`. In particular, the example shows how we can select among a pool of heterogenoeus alternatives that includes 2D convolutions with different kernel-sizes, Depthwise-Separable convolutions and even the possibility to completely skip the layer (i.e., `nn.Identity`).

Then, similarly to [PIT](../pit/README.md#basic-usage) we need to follow three steps to build an searchable network, perform search and finally export the discovered net.

1. Import the `PITSuperNet` conversion module and use it to automatically convert your `model` in an optimizable format. In its basic usage, `PITSuperNet` requires as arguments:
    - the `model` to be optimized
    - the `input_shape` of an input tensor (without batch-size)
    - the `regularizer` to be used (consult [supported regularizers](#supported-regularizers) to know the different alternatives) which dictates the metric that will be optimized.
    ```python
    from plinio.methods import PITSuperNet

    net = SimpleSuperNet()
    pitsn_net = PITSuperNet(net, input_shape=input_shape, regularizer='macs')
    ```
2. Inside the training loop compute regularization-loss and add it to task-loss to optimize the two quantities together. N.B., we suggest to control the relative balance between the two losses by multiplying a scalar `strength` value to the regularization loss.
    ```python
    strength = 1e-6  # depends on user specific requirements
    for epoch in range(N_EPOCHS):
        for sample, target in data:
            output = pitsn_net(sample)
            task_loss = criterion(output, target)
            reg_loss = strength * pitsn_net.get_regularization_loss()
            loss = task_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ```
3. Finally export the optimized model. After conversion we suggest to perform some additional epochs of fine-tuning on the `exported_model`.
    ```python
    exported_model = pitsn_net.arch_export()
    ```

### Avoid PIT-like optimization in SuperNet
As said, `plinio.pit_supernet` combines a supernet-based approach with `plinio.pit`.

Nevertheless, if the user want to avoid the PIT-like optimization and only performing an exploration based on the defined layer alternatives it is sufficient to pass the argument `autoconvert_layers=False` to `PITSuperNet`.

## Supported Regularizers
At the current state the following regularization strategies are supported:
- **Size**: this strategy tries to reduce the total number of parameters of the target layers. It can be used by specificying the argument `regularizer` of `PIT()` with the string `'size'`.
- **MACs**: this strategy tries to reduce the total number of operations of the target layers. It can be used by specificying the argument `regularizer` of `PIT()` with the string `'macs'`.
