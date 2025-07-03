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
wlim = (1e-4, 1e4)  # frequency range for plotting
plim = (0, 100)  # parameter interval
n_ps = (8, 64)  # numbers of parametric samples
loewner_opts = {
    'partitioning': 'even-odd', # loewner partitioning ('even-odd', 'half-half', 'same')
    'ordering': 'regular',  # criterion w.r.t. which the splitting is done ('regular' 'magnitude', 'random')
    'mimo_handling': 'full',
    'conjugate': False
}
r = None # if r is None, cutoff is used to determine rank
loewner_tol = 1e-7  # truncation of singular values
cond_tol = 1e42  # threshold for switching the computation formulas
name = 'rational'

#====================================================================================================
# INTERPOLATION

sys = load_example(name)
for n_p in n_ps:
    p = np.concatenate([mu['p'] for mu in sys.parameters.space(plim).sample_uniformly(n_p)])
    kinds = ('linear', 'quadratic', 'cubic', 'barycentric', 'rational')
    for kind in kinds:
        if kind == 'rational':
            rom, tf_itpl = rational_interpolator(p, sys, r=r, tol=loewner_tol, cond_tol=cond_tol, loewner_opts=loewner_opts)
        else:
            rom, tf_itpl = None, interpolator(p, sys, kind=kind)
        plotname = name + '-' + kind
        plt.close('all')
        plot_2derr(sys.transfer_function, tf_itpl, wlim, plim, rom=rom, tol=cond_tol, path=Path(name) / str(n_p), kind=kind)
