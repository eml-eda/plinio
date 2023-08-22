# PIT - Pruning in Time

## Overview
PIT is a mask-based DNAS tool capable to optimize the most relevant hyper-parameters of CNNs. Namely, it can perform channel pruning, removing output channels from both 1D and 2D Convolutional layers. Furthermore, it can also optimize the receptive field and dilation of 1D convolutions.
For the algorithm's technical details, please refer to our publication: [Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge](https://ieeexplore.ieee.org/abstract/document/9782512).

## Basic Usage
To optimize a DNN with `plinio.pit`, the minimum requirement is to add **three additional steps** to a normal PyTorch training loop:
1. Import the `PIT` class and use it to automatically convert a CNN `model` in an optimizable format. In its basic usage, `PIT` requires three arguments:
    - the `model` to be optimized
    - the `input_shape` of an input tensor (without batch-size), or alternatively, an input_example, i.e., a single input tensor. Those are required for internal shape propagation.
    - The `cost` model(s) to be used. See [cost models](../../cost/README.md) for details on the available cost models.

```python
from plinio.methods import PIT
pit_model = PIT(model, input_shape=input_shape, cost=params)
```

2. In the training loop, compute the model cost and add it to the standard task-dependent loss to co-optimize the two quantities. Since the value ranges of the two terms might be very different depending on your selected DNN and task, it is important to control the relative balance between the two loss terms by means of a scalar `strength` value:

```python
strength = 1e-6  # depends on user specific requirements
for epoch in range(N_EPOCHS):
    for sample, target in data:
        output = pit_model(sample)
        task_loss = criterion(output, target)
        loss = task_loss + strength * pit_model.cost
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

3. Finally, export the optimized model. After conversion, you normally want to perform some additional epochs of fine-tuning on the `exported_model`.

```python
exported_model = pit_model.export()
```

### Exclude layers from the optimization
In general, when `PIT` is applied to a network all the [supported layers](#supported-layers) are automatically marked as optimizable. To give maximum flexibility to the user, `PIT` also permits to exclude layers from the optimization process.

In particular, the `PIT` class constructor exposes two optional arguments that can be used to this end:

1. `exclude_names`, which expects a tuple of layer names that we want to exclude from the optimization. For instance, in the code below the `Linear` layers `lin1` and `lin2` will **not** be optimized:

```python
class Net(nn.Module):
    def __init__(self):
        self.c0 = nn.Conv1d()
        self.lin0 = nn.Linear()
        self.lin1 = nn.Linear()
        self.lin2 = nn.Linear()

net = Net()
exclude_names = ('net.lin1', 'net.lin2')
pit_net = PIT(net, input_shape=input_shape, cost=params, exclude_names=exclude_names)
```

2. `exclude_types`, which expects a tuple of layer *class names* that we want to exclude. All the layers of those types will **not** be optimized. In the code below, all `nn.Conv1d` layers will be exluded from the optimization (only `c0` for this particular DNN):

```python
class Net(nn.Module):
    def __init__(self):
        self.c0 = nn.Conv1d()
        self.lin0 = nn.Linear()
        self.lin1 = nn.Linear()
        self.lin2 = nn.Linear()

net = Net()
exclude_types = (nn.Conv1d)
pit_net = PIT(net, exclude_types=exclude_types)
```

### Optimize only specific layers
Conversely, it may happens that a user only wants to optimize few specific layers of their DNN. In this case, we give the possibility to the user to directly define and use the optimizable version of such layers in their DNN definition code.  In particular, the user can directly instantiate layers defined in `plinio.methods.pit.nn` as drop-in replacement for standard `nn.Module` sub-classes. For example:

```python
from plinio.methods.pit.nn import PITConv1d

class Net(nn.Module):
        def __init__(self):
            self.c0 = nn.PITConv1d()
            self.lin0 = nn.Linear()
            self.lin1 = nn.Linear()
            self.lin2 = nn.Linear()

    net = Net()
    pit_net = PIT(net, autoconvert_layers=False)
```
In this case, *only* `c0` will be optimized, given that we also set `autoconvert_layers` flag to `False` in the `PIT` constructor, to disable automatic replacement of [supported layers](#supported-layers).

## Supported Layers
At the current state the optimization of the following layers is supported with the PIT algorithm:
|Layer   | Hyper-Parameters  |
|:-:|:-:|
| Conv1d  | Output-Channels, Kernel-Size, Dilation |
| Conv2d  | Output-Channels  |
| Linear  | Output-Features  |

## Known Limitations

TBD