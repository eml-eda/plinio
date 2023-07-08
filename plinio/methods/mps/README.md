# Mixed Precision Assignement (Experimental)

## Overview
`plinio.mixprec` offers a DNAS-based tool to automatically explore and assign the quantization precision to different parts of the provided network. In particular, `plinio.mixprec` is able to explore and assign independent precisions to *weights* and *activations* of convolutional layers.

`plinio.mixprec` implements two different schemes:

1. **Per-Layer** Mixed-Precision Search. Is the default scheme, supported for both activations and weights. With Per-Layer scheme an indipendent precision is selected for **all* activations and **all* weights of a convolutional layer.
2. **Per-Channel** Mixed-Precision Search. Currently supported **only for weights**. With Per-Channel scheme an indipendent precision is selected for **each* different **output channel* of the weight tensor of a convolutional layer.

In order to gain more technical details about the algorithm, please refer to our publication [Channel-wise Mixed-precision Assignment for DNN Inference on Constrained Edge Nodes
](https://ieeexplore.ieee.org/abstract/document/9969373).

**N.B., the method is still experimental and poorly tested.
Additionaly, only the Per-Layer scheme is currently supported**

## Basic Usage
To optimize your model with `plinio.mixprec` you will need in most situations only **three additional steps** with respect to a normal pytorch training loop:
1. Import the `MixPrec` conversion module and use it to automatically convert your `model` in an optimizable format. In its basic usage, `MixPrec` requires as arguments:
    - the `model` to be optimized
    - the `input_shape` of an imput tensor (without batch-size)
    - the tuple of `activation_precisions` that we want to explore
    - the tuple of `weight_precisions` that we want to explore
    - the `w_mixprec_type`, i.e., the mixprec scheme that we want ot use for weights. It could be `MixPrecType.PER_LAYER` or `MixPrecType.PER_CHANNEL` (default: `MixPrecType.PER_LAYER`)
    - the `regularizer` to be used (consult [supported regularizers](#supported-regularizers) to know the different alternatives) which dictates the metric that will be optimized.
    ```python
    from plinio.methods import MixPrec
    a_prec = (2, 4, 8)
    w_prec = (2, 4, 8)
    mixprec_model = MixPrec(model,
                            input_shape=input_shape,
                            activation_precisions=a_prec,
                            weight_precisions=w_prec,
                            regularizer='size')
    ```
2. Inside the training loop compute regularization-loss and add it to task-loss to optimize the two quantities together. N.B., we suggest to control the relative balance between the two losses by multiplying a scalar `strength` value to the regularization loss.
    ```python
    strength = 1e-6  # depends on user specific requirements
    for epoch in range(N_EPOCHS):
        for sample, target in data:
            output = mixprec_model(sample)
            task_loss = criterion(output, target)
            reg_loss = strength * mixprec_model.get_regularization_loss()
            loss = task_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ```
3. Finally export the optimized model. After conversion we suggest to perform some additional epochs of fine-tuning on the `exported_model`.
    ```python
    exported_model = mixprec_model.arch_export()
    ```

## Supported Regularizers
At the current state the following regularization strategies are supported:
- **Size**: this strategy tries to reduce the total number of parameters of the target layers. It can be used by specificying the argument `regularizer` of `MixPrec()` with the string `'size'`.
- **MACs**: this strategy tries to reduce the total number of operations of the target layers. It can be used by specificying the argument `regularizer` of `MixPrec()` with the string `'macs'`.

## Supported Layers
At the current state the optimization of the following layers is supported:
|Layer   |
|:-:|
| Conv2d  |