# SuperNet

## Overview
The `plinio.supernet` sub-package implements a coarse-grained DNAS tool based on the SuperNet idea, and inspired by [DARTS](https://arxiv.org/abs/1806.09055).

A SuperNet is a network in which some of the layers include a pool of alternative operations. During training, the DNAS learns to select from each pool the operation that better balances accuracy and cost. A common usage example is the exploration of the trade-offs offered by *standard multi-channel convolutions* versus *depthwise-separable convolutions* in CNNs. Using an identity operation as one of the alternatives in the pool also enables SuperNet to explore the DNN depth (i.e., skipping one or more layers).

## Basic Usage
With respect to [PIT](../pit/README.md#basic-usage) and [MPS](../mps/README.md#basic-usage), the `plinio.supernet` API has one key difference, which is that the user is required to "manually" define the alternatives to be explored for each layer. Apart from that, the API is similar to the other two methods, as shown in the example below.

#### Search-Space Limitations
At the current stage, `plinio.supernet` requires to define a pool of layer alternatives sharing **the same output tensor dimension**.

### Example of SuperNet definition and usage
The API provided by `plinio` to define the pool of sub-networks among which the DNAS should select an alternative consists of a special `nn.Module` sub-class called `SuperNetModule`. The constructor for this class takes as input a list of standard `nn.Module` instances, each corresponding to one alternative in the pool. Once constructed, `SuperNetModule` instances can be used as drop-in replacements for standard `nn.Module` instances in your DNN definition. For instance, we may define a minimal SuperNet as follows:
```python
from torch import nn
from plinio.methods.supernet import SuperNetModule

class MyNN(nn.Module):
    def __init__(self):
        self.conv = SuperNetModule([
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
In this example, `plinio.supernet` will select between using: i) a single 2D Convolutional layer with 3x3 kernel, ii) two such layers in sequence, iii) a 2D Conv. with 5x5 kernel, and iv) an Identity operation (i.e., no-operation).

Then, similarly to [PIT](../pit/README.md#basic-usage) and [MPS](../mps/README.md#basic-usage) we need to follow three steps to optimize the DNN:

1. Import the `SuperNet` class and use it to automatically convert your PyTorch `model` into an optimizable format. In its basic usage, `SuperNet` requires three arguments:
    - The `model` to be optimized
    - The `input_shape` of an input tensor (without batch-size), or alternatively, an `input_example`, i.e., a single input tensor. Those are required for internal shape propagation.
    - The `cost` model(s) to be used. See [cost models](../../cost/README.md) for details on the available cost models.

```python
from plinio.methods import SuperNet

net = NyNN()
net = SuperNet(net, input_shape=input_shape, cost=params)
```
2. In the training loop, compute the model cost and add it to the standard task-dependent loss to co-optimize the two quantities. Since the value ranges of the two terms might be very different depending on your selected DNN and task, it is important to control the relative balance between the two loss terms by means of a scalar `strength` value:

```python
strength = 1e-6  # depends on user specific requirements
for epoch in range(N_EPOCHS):
    for sample, target in data:
        output = net(sample)
        loss = task_loss + strength * net.cost
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
3. Finally, export the optimized model. After conversion, you normally want to perform some additional epochs of fine-tuning on the exported `net`.

```python
net = net.export()
```

## Known Limitations

TBD