#!/usr/bin/env python3

import numpy as np
import scipy.interpolate as spin

from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.loewner import LoewnerReductor

from utils import assemble_ptf, plti_to_tf


def rational_interpolator(p, plti, r=None, tol=None, cond_tol=1e6, loewner_opts={}):
    sys = plti_to_tf(plti)
    loewner = LoewnerReductor(p, sys, **loewner_opts)
    rom = loewner.reduce(r=r, tol=tol)
    L = loewner.loewner_quadruple()[0]
    print(f'Dimension of IL: {L.shape[0]}x{L.shape[1]}')
    print(f'Loewner truncation rank: {rom.order}')

    # 2) compute the system matrices
    UTIV, IWV = rom.B.matrix, rom.C.matrix
    A = rom.A.matrix
    Ep = rom.E.matrix
    B = UTIV[:, -plti.dim_input:]
    C = IWV[-plti.dim_output:]
    X = IWV[:-plti.dim_output]
    Y = UTIV[:, :-plti.dim_input]

    # 3) assemble the transfer function
    return assemble_ptf(Ep, A, B, C, X, Y, plti.parameters, tol=cond_tol)


def interpolator(p, plti, kind=None):
    n = plti.order
    # convert parametric model to large transfer function, i.e. H(p) = [[A(p), B(p)], [C(p), D(p)]]
    sys = plti_to_tf(plti)
    H = sys.freq_resp(p/1j)
    if kind in ('linear', 'quadratic', 'cubic'):
        itpl = spin.interp1d(p, np.real(H), kind=kind, axis=0)
    elif kind == 'barycentric':
        itpl = spin.BarycentricInterpolator(p, H)

    def tf(s, mu=None):
        p = mu['p'][0]
        H = itpl(p)
        return LTIModel.from_matrices(H[:n, :n], H[:n, n:], H[n:, :n], H[n:, n:]).transfer_function.eval_tf(s)

    return TransferFunction(dim_input=plti.dim_input, dim_output=plti.dim_output, tf=tf, parameters=plti.parameters)
