# FlexNAS

A flexible library for testing and comparing different (differentiable) NAS techniques

Folders structure:
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