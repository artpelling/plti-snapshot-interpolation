#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from pymor.reductors.loewner import LoewnerReductor

from utils import (
    assemble_ptf,
    load_example,
    parametric_lti_to_tf,
    plot_2derr,
    plot_mags,
    plot_spectra,
)

#====================================================================================================
# VARIABLES
plim = (0, 100)  # parameter interval
n_p = 4  # number of parametric samples
partitioning = 'even-odd'  # loewner partitioning ('even-odd', 'half-half', 'same')
ordering = 'regular'  # criterion w.r.t. which the splitting is done ('regular' 'magnitude', 'random')
r = None  # if r is None, cutoff is used to determine rank
loewner_tol = 1e-7  # truncation of singular values
cond_tol = 1e6  # threshold for switching the computation formulas

#====================================================================================================
# INITIALIZATION

name = 'toy'
sys = load_example(name)
# convert parametric model to large transfer function, i.e. H(p) = [[A(p), B(p)], [C(p), D(p)]]
tf = parametric_lti_to_tf(sys)


#===================================================================================================
# ALGORITHM

# 1) sample H(p) and interpolate samples with the Loewner framework
p = np.concatenate([mu['p'] for mu in sys.parameters.space(plim).sample_uniformly(n_p)])  # sample parameter space
loewner = LoewnerReductor(p, tf, partitioning=partitioning, ordering=ordering, mimo_handling='full', conjugate=False)
rom = loewner.reduce(tol=loewner_tol)
L = loewner.loewner_quadruple()[0]
print(f'Dimension of IL: {L.shape[0]}x{L.shape[1]}')
print(f'Loewner truncation rank: {rom.order}')

# 2) compute the system matrices
UTIV, IWV = rom.B.matrix, rom.C.matrix
A = rom.A.matrix
Ep = rom.E.matrix
B = UTIV[:, -sys.dim_input:]
C = IWV[-sys.dim_output:]
X = IWV[:-sys.dim_output]
Y = UTIV[:, :-sys.dim_input]

# 3) assemble the transfer function
tf_itpl = assemble_ptf(Ep, A, B, C, X, Y, sys.parameters, tol=cond_tol)


#===================================================================================================
# PLOTTING
plt.close('all')

wlim = (1e-2, 1e8)  # frequency range for plotting
n_plot = 10  # number of parameters for plotting
P = sys.parameters.space(plim).sample_uniformly(2*n_plot+1)[1::2]
plot_mags(sys.transfer_function, tf_itpl, P, (1, 300), name=name)
IL, ILs = loewner.loewner_quadruple()[:2]
plot_spectra(IL, ILs, rom.order, 2*n_p, name=name)
plot_2derr(sys.transfer_function, tf_itpl, Ep, A, X, Y, wlim, plim, tol=cond_tol, name=name)
