#!/usr/bin/env python3

import numpy as np
import scipy.interpolate as spin

from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.loewner import LoewnerReductor

from utils import estimate_rcond, plti_to_tf


def rational_interpolator(p, plti, r=None, tol=None, cond_tol=1e6, loewner_opts={}):
    sys = plti_to_tf(plti)
    loewner = LoewnerReductor(p, sys, **loewner_opts)
    rom = loewner.reduce(r=r, tol=tol)
    L = loewner.loewner_quadruple()[0]
    print(f'Dimension of IL: {L.shape[0]}x{L.shape[1]}')
    print(f'Loewner truncation rank: {rom.order}')

    def tf(s, mu=None):
        p = mu['p'][0]
        K = p*rom.E.matrix - rom.A.matrix
        Zsp = K - 1/s * (rom.B.matrix[:, :-plti.dim_input] @ rom.C.matrix[:-plti.dim_output])
        rcond = estimate_rcond(Zsp)
        if rcond == 0 or cond_tol > 1/rcond:
            return rom.C.matrix[-plti.dim_output:] @ np.linalg.inv(Zsp) @ rom.B.matrix[:, -plti.dim_input:]
        else:
            H = rom.transfer_function.eval_tf(p)
            return LTIModel.from_matrices(H[:plti.order, :plti.order], H[:plti.order, plti.order:], H[plti.order:, :plti.order], H[plti.order:, plti.order:]).transfer_function.eval_tf(s)

    return rom, TransferFunction(dim_input=plti.dim_input, dim_output=plti.dim_output, tf=tf, parameters=plti.parameters)


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
