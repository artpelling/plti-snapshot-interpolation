#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from interpolation import interpolator, rational_interpolator
from utils import (
    load_example,
    plot_2derr,
)

#====================================================================================================
# VARIABLES
plim = (1, 100)  # parameter interval
n_p = 8  # number of parametric samples
loewner_opts = {
    'partitioning': 'even-odd', # loewner partitioning ('even-odd', 'half-half', 'same')
    'ordering': 'regular',  # criterion w.r.t. which the splitting is done ('regular' 'magnitude', 'random')
    'mimo_handling': 'full',
    'conjugate': False
}
r = None # if r is None, cutoff is used to determine rank
loewner_tol = 1e-12  # truncation of singular values
cond_tol = 1e48  # threshold for switching the computation formulas
name = 'rational'
path = Path(name) / str(n_p)  # path for saving plots

#====================================================================================================
# INITIALIZATION

sys = load_example(name)
# convert parametric model to large transfer function, i.e. H(p) = [[A(p), B(p)], [C(p), D(p)]]

#===================================================================================================
# ALGORITHM
# 1) compute samples of H(p)
p = np.concatenate([mu['p'] for mu in sys.parameters.space(plim).sample_uniformly(n_p)])

kinds = ('linear', 'quadratic', 'cubic', 'barycentric', 'rational')
for kind in kinds:
    if kind == 'rational':
        rom, tf_itpl = rational_interpolator(p, sys, r=r, tol=loewner_tol, cond_tol=cond_tol, loewner_opts=loewner_opts)
    else:
        rom, tf_itpl = None, interpolator(p, sys, kind=kind)
    plotname = name + '-' + kind

    #===================================================================================================
    # PLOTTING
    plt.close('all')
    wlim = (1e-4, 1e4)  # frequency range for plotting
    plot_2derr(sys.transfer_function, tf_itpl, wlim, plim, rom=rom, tol=cond_tol, path=path, kind=kind)
