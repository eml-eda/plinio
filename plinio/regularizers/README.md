# PLiNIO Regularizers

## Overview
The `plinio.regularizers` sub-package contains the definition of different regularizers for cost-aware DNN optimization. Regularizers are essentially ways to combine one or more cost models (see [here](../cost/README.md)) to form a PyTorch loss function component. Similarl to the philosophy used when defining cost models, regularizers are *orthogonal* to the selected gradient-based optimization method. In practice, all regularizers can be used with any of the optimization algorithms supported by PLiNIO.

## Supported Regularizers

The current version of PLiNIO supports the following regularizers:
* `BaseRegularizer`, an implementation of the standard single-cost-aware DNAS regularizer (task_loss + strength * cost).
* `DUCCIO`, an implementation of the "DNAS Under Combined Constraints In One-host", proposed in [this paper](https://arxiv.org/abs/2310.07217)

### Base Regularizer

The `BaseRegularizer` is nothing but a wrapper to the standard cost-aware DNAS formulation.

An instance of this class is consturcted as follows:

```python
BaseRegularizer(cost_name=<cost_name>, strength=<strength>)
```

where `cost_name` is the name of a single DNN cost metric, and `strength` is a real-valued, constant regularization strength. The cost metric name should be the same used when defining the cost specification of a PLiNIO model. Applying the constructed instance to a PLiNIO model retrieves the current cost metric value, multiplies it with the regularization strength factor and returns it.

The following code is a basic usage example of this regularizer with the [PIT](../methods/pit/README.md) optimization method:
```python
from plinio.cost import params
from plinio.regularizers import BaseRegularizer

model = PIT(model, input_shape=input_shape, cost={'params': params})
regularizer = BaseRegularizer('params', 1e-3)

# ... other code ...

# training loop
for epoch in range(N_EPOCHS):
    for sample, target in data:
        output = model(sample)
        # the next line is equivalent to:
        # criterion(output, target) + 1e-3 * model.get_cost('params')
        loss = criterion(output, target) + regularizer(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### DUCCIO

The `DUCCIO` regularizer implements the multi-constraint regularization scheme described in [this paper](https://arxiv.org/abs/2310.07217). Differently from `BaseRegularizer` this method treats cost metrics as *constraints*, rather than secondary objectives. Thus, models are only penalized when a specific cost metric exceeds a user-defined *target*. The constructor is as follows:

```python
DUCCIO(targets, <task_loss/final_strengths>)
```

where `targets` is a (str, tensor) dictionary, containing the names of the cost metrics to be considered, and the corresponding target values.
To follow the implementation of the paper, users shall also provide the `DUCCIO` constructor with the value of the task loss at the end of the warm up phase (i.e., the training of the original model), through the `task_loss` parameter, which accepts a scalar tensor-like object. From this value, the regularizer derives the final regularization strength value to be used for each considered metric, using Eq. 13 of the paper.

The regularization should be linearly increased at each epoch to reach the desired value at the end of the search phase, following Eq. 14 of the paper. This proved more effective than using a fixed strength, in order to obtain a model that meets all constraints while preserving accuracy as much as possible. Alternatively, users can also specify final strengths manually, through the `final_strengths` constructor parameter, which accepts a tuple with the same number of items as `targets`. Exactly one between `task_loss` and `final_strengths` should be set in the constructor.

In order to easily achieve the required regularization strength scheduling, applying the DUCCIO regularizer to a PLiNIO model requires passing two additional parameters to the call method, corresponding to the current epoch number (`epoch`) and to the total number of epochs foreseen for the search phase (`n_epochs`). Vice versa, to use the final strength values directly, without scheduling, the two parameters can be left unspecified.

The following code is a basic example of using DUCCIO with the [PIT](../methods/pit/README.md) optimization method and two cost constraints (in terms of model size and number of OPs):

```python
from plinio.cost import params, ops
from plinio.regularizers import DUCCIO

wu_task_loss = # final training set loss after the warmup phase (training of the original model)

model = PIT(model, input_shape=input_shape, cost={'params': params, 'ops': ops})
# look for models with less than 20k parameters and 4M OPs
regularizer = DUCCIO({'params': 2e+4, 'ops': 4e+6}, task_loss=wu_task_loss)

# ... other code ...

# training loop
for epoch in range(N_EPOCHS):
    for sample, target in data:
        output = model(sample)
        loss = criterion(output, target) + regularizer(model, epoch, N_EPOCHS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

