# PIT - Pruning in Time

## Overview
PIT is a mask-based DNAS tool capable to optimize the most relevant hyper-parameters of 1D and 2D CNNs balancing performance-task with other metrics denoted with in-general as *regularization*.
The list of the supported regularization targets is available in section [Supported Regularizers](#supported-regularizers).

In order to gain more technical details about the algorithm, please refer to our publication [Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge](https://ieeexplore.ieee.org/abstract/document/9782512).

## Usage
To optimize your model with PIT you will need in most situations only **three additional steps** with respect to a normal pytorch training loop:
1. Import the `PIT` conversion module and use it to automatically convert your `model` in an optimizable format. In its basic usage, `PIT` requires as arguments:
    - the `model` to be optimized
    - the `input_shape` of an imput tensor (without batch-size)
    - the `regularizer` to be used (i.e., `'size'` or `'macs'`) which dictates the metric that will be optimized.
    ```python
    from plinio import PIT
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

To see an end-to-end example you can consult the **TODO Tutorial**.

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