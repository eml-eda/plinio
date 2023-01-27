# PIT - Pruning in Time

## Overview
PIT is a mask-based DNAS tool capable to optimize the most relevant hyper-parameters of 1D and 2D CNNs balancing performance-task with other metrics denoted with in-general as *regularization*.
The list of the supported regularization targets is available in section [Supported Regularizers](#supported-regularizers).

In order to gain more technical details about the algorithm, please refer to our publication [Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge](https://ieeexplore.ieee.org/abstract/document/9782512).

## Basic Usage
To optimize your model with PIT you will need in most situations only **three additional steps** with respect to a normal pytorch training loop:
1. Import the `PIT` conversion module and use it to automatically convert your `model` in an optimizable format. In its basic usage, `PIT` requires as arguments:
    - the `model` to be optimized
    - the `input_shape` of an imput tensor (without batch-size)
    - the `regularizer` to be used (consult [supported regularizers](#supported-regularizers) to know the different alternatives) which dictates the metric that will be optimized.
    ```python
    from plinio.methods import PIT
    pit_model = PIT(model, input_shape=input_shape, regularizer='size')
    ```
2. Inside the training loop compute regularization-loss and add it to task-loss to optimize the two quantities together. N.B., we suggest to control the relative balance between the two losses by multiplying a scalar `strength` value to the regularization loss.
    ```python
    strength = 1e-6  # depends on user specific requirements
    for epoch in range(N_EPOCHS):
        for sample, target in data:
            output = pit_model(sample)
            task_loss = criterion(output, target)
            reg_loss = strength * pit_model.get_regularization_loss()
            loss = task_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ```
3. Finally export the optimized model. After conversion we suggest to perform some additional epochs of fine-tuning on the `exported_model`.
    ```python
    exported_model = pit_model.arch_export()
    ```

### Exclude layers from the optimization
In general, when `plinio.PIT` is applied to a network all the [supported layers](#supported-layers) are automatically marked as optimizable.

In the spirit of giving maximum flexibility to the user, `plinio.PIT` allows to exclude layers from the optimization process.

In particular, `plinio.PIT` exposes two arguments that can be used with this aim:
1. `exclude_names`, is an optional tuple of layer identifiers that we want to exclude. Only the specified layers will **not** be optimized. E.g.,
    ```python
    class Net(nn.Module):
        def __init__(self):
            self.c0 = nn.Conv1d()
            self.lin0 = nn.Linear()
            self.lin1 = nn.Linear()
            self.lin2 = nn.Linear()

    net = Net()
    exclude_names = ('net.lin1', 'net.lin2')
    pit_net = plinio.PIT(net, exclude_names=exclude_names)
    ```
    In the example, the `Linear` layers `lin1` and `lin2` will **not** be optimized.

2. `exclude_types`, is an optional tuple of layer types that we want to exclude. All the layers of this type will **not** be optimized. E.g.,
    ```python
    class Net(nn.Module):
        def __init__(self):
            self.c0 = nn.Conv1d()
            self.lin0 = nn.Linear()
            self.lin1 = nn.Linear()
            self.lin2 = nn.Linear()

    net = Net()
    exclude_types = (nn.Conv1d)
    pit_net = plinio.PIT(net, exclude_types=exclude_types)
    ```
    In the example, all the `nn.Conv1d` will **not** be optimized. I.e., the layer `c0` will be excluded from the optimization process.


### Optimize only specific layers
Conversely, it may happens that we would to optimize only specific layers of our network.
In this case, we give the possibility to the user to directly define and use the optimizable version of such layers in the net.

In particular, the user will use the layers defined in `plinio.pit.nn`. E.g.,

```python
from plinio.pit.nn import PITConv1d

class Net(nn.Module):
        def __init__(self):
            self.c0 = nn.PITConv1d()
            self.lin0 = nn.Linear()
            self.lin1 = nn.Linear()
            self.lin2 = nn.Linear()

    net = Net()
    pit_net = plinio.PIT(net, autoconvert_layers=False)
```
In this example, only the layer `c0` will be optimized.

Please note that in this case we need to specify the `autoconvert_layers=False` argument to `plinio.PIT` to tell that we do **not** want to automatically convert all [supported layers](#supported-layers).

## Supported Regularizers
At the current state the following regularization strategies are supported:
- **Size**: this strategy tries to reduce the total number of parameters of the target layers. It can be used by specificying the argument `regularizer` of `PIT()` with the string `'size'`.
- **MACs**: this strategy tries to reduce the total number of operations of the target layers. It can be used by specificying the argument `regularizer` of `PIT()` with the string `'macs'`.

## Supported Layers
At the current state the optimization of the following layers are supported:
|Layer   | Hyper-Parameters  |
|:-:|:-:|
| Conv1d  | Output-Channels, Kernel-Size, Dilation |
| Conv2d  | Output-Channels  |
| Linear  | Output-Features  |