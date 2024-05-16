# MPS - Mixed Precision Search

## Overview
`plinio.mps` implements a gradient-based tool to automatically explore and assign an integer quantization precision to different parts of a DNN. In particular, `plinio.mps` is able to explore and assign independent precisions to *weights* and *activations* of convolutional and linear layers. Moreover, when it is applied with a single precision choice, it can be used to implement a standard Quantization-Aware Training (QAT).

The precision assignment in `plinio.mps` can have two granularity levels:

1. **Per-Layer**: the default scheme, supported for both activations and weights. This scheme assigns a single precision to the *entire* activations and weights tensors of each layer.
2. **Per-Channel**: currently supported **only for weights**. With this scheme, an indipendent precision is selected for **each output channel** of the weight tensor of a convolutional layer. Importantly, this second scheme also supports using **0-bit precision** for some of the channels, thus implementing a joint channel-pruning and MPS scheme.

For the technical details on this optimization algorithm, please refer to our publication: [Channel-wise Mixed-precision Assignment for DNN Inference on Constrained Edge Nodes
](https://ieeexplore.ieee.org/abstract/document/9969373).

**Important:**: currently, the `export()` API for exporting the final optimized model crashes for the per-channel assignment scheme. This will be fixed in a future release.

## Basic Usage
To optimize a DNN with `plinio.mps`, the minimum requirement is to add **three additional steps** to a normal PyTorch training loop:

1. Import the `MPS` class and use it to automatically convert your `model` in an optimizable format. In its basic usage, the `MPS` constructor requires the following arguments:
    - the `model` to be optimized
    - The `input_shape` of an input tensor (without batch-size), or alternatively, an `input_example`, i.e., a single input tensor. Those are required for internal shape propagation.
    - A `qinfo` dictionary, containing information about the quantization scheme to be used by the NAS, including the precision values to be considered in the search, as well as the type of quantizer, and additional optional parameters. The exposed `get_default_qinfo()` function can be used to quickly get some good defaults, while customizing the search precision (for all layers) through two tuples of integers passed as parameters. Alternatively, the dictionary can be modified by hand to specify more advanced quantization schemes.
    - The selected precision assignment scheme for the weights (`w_search_type`),  which can be either `MPSType.PER_LAYER` or `MPSType.PER_CHANNEL` (default: `MPSType.PER_LAYER`)
    - The `cost` model(s) to be used. See [cost models](../../cost/README.md) for details on the available cost models.

```python
from plinio.methods.mps import MPS, MPSType, get_default_qinfo
model = MPS(model,
            input_shape=input_shape,
            qinfo=get_default_qinfo(
                w_precision=(2, 4, 8),
                a_precision=(2, 4, 8)),
            w_search_type=MPSType.PER_LAYER,
            cost=params_bit)
```

2. In the training loop, compute the model cost and add it to the standard task-dependent loss to co-optimize the two quantities. Since the value ranges of the two terms might be very different depending on your selected DNN and task, it is important to control the relative balance between the two loss terms by means of a scalar `strength` value:

```python
strength = 1e-6  # depends on user specific requirements
for epoch in range(N_EPOCHS):
    for sample, target in data:
        output = model(sample)
        task_loss = criterion(output, target)
        loss = task_loss + strength * model.cost
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

3. Finally export the optimized model. export the optimized model. After conversion, you normally want to perform some additional epochs of fine-tuning on the exported `model`.

```python
model = model.export()
```

## Supported Layers
At the current state the optimization of the following layers is supported:

|Layer   | Hyper-Parameters  |
|:-:|:-:|
| Conv2d  | Weights Precision, Activations Precision |
| Linear  | Weights Precision, Activation Precision |


## Known Limitations

TBD
