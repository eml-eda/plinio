# FlexNAS

A flexible library for testing and comparing different (differentiable) NAS techniques

## Folders structure:
- `flexnas`: main library code
  * `methods`: NAS code
    * `dnas_base`: common code for all DNASes
    * `pit`: PIT-specific code
      * `pit_model.py`: class for PIT NAS models
      * `pit_conv1d.py`: class defining a PIT Conv1D layer (template for other layers)
      * `pit_binarizer.py`: the elementary STE-based mask binarizer for all PIT masks
      * `pit_XXX_masker.py`: different type of mask layers for PIT
  * `utils`: other utility code
    * `features_calculator.py`: different classes to compute the number of features in a layer (see PITModel class)
    * `model_graph.py`: utility functions to process a model graph
    * `model_info.py`: utility functions to get information about a model
- `unit_test`: unit tests
  - `models`: some DNNs used to test the library
  - `test_methods`: unit tests for DNAS methods
  - `test_utils`: unit tests for utility functions
- `benchmarks`: sub-module importing the main benchmarks


## ToDo List (general):

1. Test PIT-1D functionality:
    - See list of tests
2. Extend to Conv2D and Linear (adding new tests as required)
3. Add final model export functionality (+ tests)
4. Run on all 4 TinyMLPerf Benchmarks

## List of Tests:
1. PIT conversion (already started):
    1. Add more strict checks on conversion correctness (using small toy-models, e.g. one residual block):
        1. Is the input features calculator set correctly for all layers?
        1. Is the shared masker set correctly for all layers?
    1. Test conversion starting from a model that already includes `PITConv1D` layers (with and without `autoconvert_layers`)
    1. Verify that `exclude_names` and `exclude_types` work correctly
1. Masking:
    1. Verify that, setting some alpha/beta/gamma values at 0/1 "by hand" in `PITConv1D` layers, the layer output is as expected (correctly masked)
1. Regularization Loss Computation:
    1. Verify that the initial value of the regularization loss is the correct one:
        1. Both "flops" and "macs"
        1. Both single layer and full model
    1. Run a single forward + backward step with pre-defined inputs and using the regularization loss as the *only* loss component, to verify that gradients are different from zero.
    1. Run multiple forward + backward steps with random inputs and using the regularization loss as the *only* loss component, to verify that all masks (alpha/beta/gamma) that are not 'fixed' go to zero as expected.
1. Layer optimization options:
    1. Verify that by setting `train_channels` to `False`, alpha gradients remain fixed at 0.
    1. Verify that by setting `train_rf` to `False`, beta gradients remain fixed at 0.
    1. Verify that by setting `train_dilation` to `False`, gamma gradients remain fixed at 0.
1. Combined loss training:
    1. Run a single training step (with pre-defined, known inputs) using both a task loss (e.g. BCE, CCE) and the regularization loss, and verify that both normal weights and NAS masks change.
    2. Repeat for different task losses and different regularization losses ('flops', 'macs')
