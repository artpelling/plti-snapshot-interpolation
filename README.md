# Code for Numerical Experiments in "Snapshot-based Rational Interpolation of Parametric Systems"

## Installation

To run the examples and create the plots, Python 3.10 and `virtualenv` is needed.

The necessary packages are listed in [`requirements.txt`](requirements.txt).
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

## Author

Art J. R. Pelling:

- affiliation: Technische Universität Berlin
- email: a.pelling@tu-berlin.de
- ORCiD: 0000-0003-3228-6069

## License

The code is published under the MIT license.
See [LICENSE](LICENSE).
