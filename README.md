# Code for Numerical Experiments in "Snapshot-driven Rational Interpolation of Parametric Systems"
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11246111.svg)](https://doi.org/10.5281/zenodo.11246111)

This repository contains code for numerical experiments reported in

> Art J. R. Pelling, Karim Cherifi, Ion Victor Gosea, Ennes Sarradj
> **Snapshot-driven Rational Interpolation of Parametric Systems**,
> [*arXiv preprint*](https://arxiv.org/abs/2406.01236),
> 2024

## Installation

To run the examples and create the plots, Python 3.13 and `virtualenv` or `uv` is needed.

The necessary packages for installation with `virtualenv` are listed in [`requirements.txt`](requirements.txt).
They can be installed in a virtual environment by, e.g.,

```bash
python3 -m virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

or by use of the [`Makefile`](Makefile) with

``` shell
make venv
```

## Running the Experiments

The numerical examples from the paper are provided as `example_*.py` Python scripts. They can also be run with `make` by

``` shell
make figures
```

which will create the virtual environment if necessary.

For usage with `uv` simply run

``` shell
uv run example_polynomial.py
uv run example_rational.py
```


## Author

Art J. R. Pelling:

- affiliation: Technische Universität Berlin
- email: a.pelling@tu-berlin.de
- ORCiD: 0000-0003-3228-6069

## License

The code is published under the MIT license.
See [LICENSE](LICENSE).
