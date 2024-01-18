# PLiNIO Cost Models

## Overview
The `plinio.cost` sub-package contains the definition of DNN cost models supported by PLiNIO. Cost models can be used in all PLiNIO optimization methods, either as secondary optimization objectives or as (soft) constraints.

One of the key design decisions in PLiNIO is the attempt to make cost models as agnostic as possible to the type of optimization being applied to the network. For example, the same `ops` cost model, which counts the number of MAC operations per inference, can be used with the [`SuperNet`](../methods/supernet/README.md) optimization method, or with [`PIT`](../methods/pit/README.md), in a fully transparent way.

There are some exceptions to this rule. For instance, the `params_bit` cost model, which counts the total number of bits required to store the DNN parameters, triggers an exception when applied to [`PIT`](../methods/pit/README.md) or [`SuperNet`](../methods/supernet/README.md), which do not optimize the bit precision of weights. While it would be possible to make the bit precision a constant (e.g. equal to float32) for these optimization methods, the fact that the user is attempting to use a bitwidth-aware model with a bitwidth-unaware optimization is probably indication of a mistake, hence we signal it. There is of course a `params` cost model that can be used with [`PIT`](../methods/pit/README.md) and [`SuperNet`](../methods/supernet/README.md) to optimize the parameters' count (ignoring their bit precision).

**IMPORTANT:** this part of the library is still in development, although we are planning to make it the core distinctive component of PLiNIO. So, stay tuned.


## Supported Cost Models

The current version of PLiNIO supports the following cost models:

Hardware unaware:
* `params`: counts the number of WEIGHTS of the model. Ignores biases, and only considers Convolutional and Linear layers, assuming all other layers (e.g. BatchNorm) have 0 parameters.
* `params_bit`: counts the number of bits required to store the WEIGHTS of the model (relevant for [`MPS`](../methods/mps/README.md) only). Ignores biases, and only considers Convolutional and Linear layers, assuming all other layers (e.g. BatchNorm) have 0 parameters.
* `params_bias`: [DEPRECATED] counts the number of PARAMETERS (weights and biases) of the model. Considers Convolutional and Linear layers, assuming all other layers (e.g. BatchNorm) have 0 parameters. This model will replace `params` (currently kept only for reproducibility reasons) and the current name will be deprecated.
* `ops`: counts the number of MAC operations per inference. Accounts for Convolutional and Linear layers only, assuming that all other layers have 0 ops.
* `ops_bit`: counts the number of "bitops" per inference (relevant for [`MPS`](../methods/mps/README.md) only). Bitops are defined as $sum_{b_w,b_x}(b_w \cdot b_x \cdot OPS_{b_w,b_x})$, where $b_w$ and $b_x$ are the possible weights and activations bit-widths, and $OPS_{b_w, b_x}$ is the number of MAC operations executed with those bit-widths as input. Accounts for Convolutional and Linear layers only, assuming that all other layers have 0 ops.

Hardware aware:
* `diana_latency`: a bit-width and spatial parallelism dependent analytical latency model for the DIANA System-on-Chip described [here](https://ieeexplore.ieee.org/document/9731716).
* `diana_energy` (TBA): a bit-width and spatial parallelism dependent analytical energy model for the DIANA System-on-Chip described [here](https://ieeexplore.ieee.org/document/9731716).
* `gap8_latency`: a latency model for the GreenWaves' GAP8 System-on-Chip described [here](https://ieeexplore.ieee.org/document/8445101).
* `gap8_energy` (TBA): an energy model for the GreenWaves' GAP8 System-on-Chip described [here](https://ieeexplore.ieee.org/document/8445101).
* `mpic_latency` (TBA): a bit-width dependent LUT-based latency model for the MPIC RISC-V processor with mixed-precision support described [here](https://arxiv.org/pdf/2010.04073.pdf).
* `mpic_energy` (TBA): a bit-width dependent LUT-based energy model for the MPIC RISC-V processor with mixed-precision support described [here](https://arxiv.org/pdf/2010.04073.pdf).
* `ne16_latency` (TBA): a bit-width dependent latency model for the NE16 accelerator described [here](https://github.com/pulp-platform/ne16).

## Using Pre-defined Cost Models

Cost models are imported from the `cost` sub-package as follows (using `params` as an example):

```Python
from plinio.cost import params
```

Then, they can be used with PLiNIO optimization methods that support them, as follows:

```Python
model = PIT(orig_model, cost=params, ...)
# or
model = SuperNet(orig_model, cost=params, ...)
```

Lastly, the differentiable cost estimation, which depends on the internals of the selected optimization method, can be retrieved at any point during a training run with:

```Python
cost = model.cost
```

The cost specification (i.e. the cost model, or set of cost models, currently being evaluated for the DNN under optimization) can be retrieved from a PLiNIO model accessing the `cost_specification` property:

```Python
cost_spec = model.cost_specification
```
This property can also be set at any time, to change the target cost model in the middle of a training/optimization run. For example

```Python
from plinio.cost import ops

model.cost_specification = ops
```

## Using Multiple Cost Models

PLiNIO supports optimizing a DNN according to **more than one cost model** simultaneously. A practical example is trying to balance latency and accuracy under a maximum model size constraint (see [this](https://dl.acm.org/doi/abs/10.1145/3531437.3539720) paper for more details on this technique).

To associate multiple cost models to a PLiNIO optimization method, it is sufficient to pass them as a dictionary at construction time. For example:
```Python
from plinio.cost import ops, params

model = PIT(orig_model, cost={'size': params, 'macs': ops}, ...)
```

Then, the cost values according to each metric can be retrieved with the `get_cost()` method, for example:

```Python
s = model.get_cost('size')
m = model.get_cost('macs')
```

Note that accessing the (unnamed) `model.cost` property will trigger an exception if a PLiNIO DNN is associated with multiple cost models.

The cost specification can still be modified mid-training as before, for instance:
```Python
model.cost_specification = {'lat': my_latency_model, 'size': params}
```


## Defining New Cost Models

PLiNIO eases the definition of new cost models for specific hardware targets.
Namely, it supports any cost model that can be expressed as a *differentiable function* of the following three sets of inputs:
1. The complete set of layer's **hyper-parameters**. For instance, for a Convolutional layer, the number of input/channels channels, the kernel size, the dilation, etc.
1. The output activations **tensor shapes**.
1. The parameters and activations **bit-widths**.

Differentiability is required since PLiNIO uses gradient-based optimization methods. In the future, we plan to also support models that depend on weight *values* (e.g., to easily model the effect of unstructured sparsity), but this is not part of the current release.

### A Simple Example

To introduce the PLiNIO cost model API we start from a simple example. The following code defines a simplistic OPs count model, specified only for 2D Conv and Linear layers (for brevity):
```Python
from plinio.cost import CostSpec
from plinio.cost.pattern import Conv2dGeneric, LinearGeneric, Conv2dDW

def conv2d_cost(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    cost = cout * (cin * k[0] * k[1] + 1) * out_shape[2] * out_shape[3]
    return cost

def dw_conv2d_cost(spec):
    cin = spec['in_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    cost = cin * (k[0] * k[1] + 1) * out_shape[2] * out_shape[3]
    return cost

def linear_cost(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    cost = cout * (cin + 1)
    return cost


my_ops = CostSpec(shared=True, default_behavior='zero')
my_ops[Conv2dGeneric] = conv2d_cost
my_ops[Conv2dDW] = dw_conv2d_cost
my_ops[LinearGeneric] = linear_cost
```

We first define three Python functions that take a single `spec` argument, storing all supported input information for the DNN graph portion under consideration, as described above (hyperparameters, tensor shapes, and bitwidths - although the latter are not used for this model). The function must return a `torch.Tensor` or convertible type and include only operations that have a corresponding (differentiable) torch implementation.

The cost model specification is then created as an instance of the `CostSpec` class. The two constructor parameters determine respectively:
* If the cost computation should be `shared` between different executions of the same DNN sub-graph, or not. This is only relevant if a network contains layers that are executed multiple times in a single inference (e.g. RNN cells). For example, MACs/OPs cost models are defined with `shared=False` (every new invocation of the layer contributes additional ops), whereas the n. of params has `shared=True`, since it does not change regardless of how many times the layer is invoked.
* The `default_behavior` triggered when parts of a DNN do not have a matching cost function. This can be either `'zero'` (assume 0 cost) or `'fail'` (raise an exception). The former is used in most cases.

Lastly, cost functions are associated to DNN patterns. Pre-defined patterns are provided by PLiNIO, but users can also define custom ones.

The rest of this section details each element of the above example.

### Cost Functions

As anticipated, a cost function is simply any torch-based differentiable function that takes as input a data structure with the information of the target layer(s), and returns a scalar.

The `spec` input is a simple Python dictionary with string keys, called `PatternSpec` throughout the library.

To make the definition of cost models as agnostic as possible of the underlying optimization method, users can retrieve layer hyper-parameters from a `PatternSpec` using the **same names** found in PyTorch layer attributes. For example, for a `nn.Conv2D` layer, the number of output channels will be called `'out_channels'`, whereas for a `nn.Linear` layer, it will be called `'out_features'`, and so on.

Output tensor shapes are stored under the `'output_shape'` key, as a tuple of scalars following PyTorch's conventional (N,C,H,W) layout.

Lastly, we use `'in_precision'`, `'w_precision'` and `'out_precision'` to specify the bit-widths for inputs, weights and outputs, for models that support them. We also feed cost functions with corresponding `'in_format'`, `'w_format'` and `'out_format'` entries to specify the dtype for each tensor (e.g. int vs float)
**[NOTE: this might change in future versions of the library].**


### Patterns

A **`Pattern`** is the DNN sub-graph to which a cost function should be applied. Currently, we only support single layers as patterns, therefore `Pattern` is just an alias for `nn.Module`. However, the goal is being able to match more complex sub-graphs in future versions of the library.

In other words, we can currently define the cost for a Convolutional layer or for a Linear layer, but not a "monolithic" cost for a Conv-BatchNorm-ReLU sequence. Another limitation is that only operations that have a direct `nn.Module` correspondance are supported. Therefore, purely functional Torch ops such as additions or activation functions cannot be associated with a cost.

### Constraints

A **`Constraint`** is essentially a boolean function returning `True` if, besides being an instance of the correct `nn.Module` sub-class, a given `Pattern` also verifies an additional set of rules.

A constraint function takes as input a `PatternSpec` and uses its content to verify that the constraints are respected. For instance, this is what permits to have distinct cost models for a standard multi-channel convolution and for a depthwise separable convolution. Here's the constraint function for matching a 2D depthwise conv:

```Python
def conv_dw_constraint(spec: PatternSpec):
    return spec['in_channels'] == spec['groups'] and spec['out_channels'] == spec['groups']
```

Once could also specify, say, a constraint, to only match Conv layers with 3x3 filters, as follows:
```Python
def conv_3_constraint(spec: PatternSpec):
    k = spec['kernel_size']
    return all([ki == 3 for ki in k])
```

### Associating Patterns, Constraints and Cost Functions

A cost model specification basically builds an association between a DNN graph *pattern* (possibly with *constraints*), and a *cost function*. In the code, this is done treating the cost specification instance as a sort of dictionary. The general scheme is:

```Python
cs = CostSpec(...)
cs[(Pattern, Constraint)] = CostFn
```

The library provides pre-defined `(Pattern, Constraint)` tuples for common layers. For instance, `Conv2dDW` is defined as:

```Python
Conv2dDW = (nn.Conv2d, conv_dw_constraint)
```

whereas `Conv2dGeneric` is:

```Python
Conv2dGeneric = (nn.Conv2d, None)
```

Constrained patterns are **always matched before unconstrained ones**. Therefore, a depthwise convolution's cost will always be the one returned by the function associated with the `Conv2dDW` tuple, and not with `Conv2dGeneric`.

If multiple constrained patterns for the same `nn.Module` sub-class are provided, the library assumes that **at most one will match**. If this is not true, it is unspecified which of the matching cost functions will be invoked.

## Known Limitations

TBD
