<div align="center">
<img src=".assets/plinio_logo.png" width="700"/>
</div>
<sub><sup><i>N.B., logo partially generated with Stable Diffusion.</i></sup></sub>

---

**PLiNIO** is a Python package built on-top of the PyTorch ecosystem that provides a **P**lug-and-play **Li**ghtweight tool for the **I**nference **O**ptimization of Deep **N**eural networks (DNNs).

PLiNIO allows to automatically optimize your DNN's architecture with ***no more than three additional lines of code*** to your original training loop.

<div align="center">
<img src=".assets/train_loop_plinio.png" width="500"/>
</div>

PLiNIO exploits as main optimization engine Differentiable Neural Architecture Search (DNAS) algorithms which notoriusly balance flexibility and lightness.

At the current state, PLiNIO implements the following methods:
- [PIT](plinio/methods/pit/README.md)
- [PIT-SuperNet](plinio/methods/pit_supernet/README.md)
- [Mixed Precision](plinio/methods/mixprec/README.md) **[N.B., this feature is experimental and currently under development]**

You can consult the specific linked pages to gather more information on the usage and the capabilieties of such tools.

# Installation
To install the latest release:

```
$ git clone https://github.com/eml-eda/plinio
$ cd plinio
$ python setup.py install
```

# License
PLiNIO entire codebase is released under [Apache License 2.0](LICENSE).
