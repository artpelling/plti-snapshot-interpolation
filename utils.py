#!/usr/bin/env python3

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from time import perf_counter

import numpy as np
import scipy.linalg as spla
from pathlib import Path

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameters
from pymor.parameters.functionals import ExpressionParameterFunctional


#====================================================================================================
# INITIALIZATION

def load_example(name):
    parameter = Parameters({'p': 1})  # create a one dimensional parameter
    if name == 'polynomial':
        """Polynomial example from https://doi.org/10.1109/CDC45484.2021.9682841 Sec. V-B"""
        A0 = NumpyMatrixOperator(np.diag([-2, -1, -1]))
        A1 = NumpyMatrixOperator(np.array([[0, -1, 0], [0, 0, -0.5], [0, -0.5, 0]]))
        A2 = NumpyMatrixOperator(np.array([[0.1, 0, 0.2], [0, 1, 0], [-0.2, 0, 0]]))
        A3 = NumpyMatrixOperator(np.array([[0, 1, 0], [-1, 0, 0], [0, -10, 0]]))
        p = ExpressionParameterFunctional(list(parameter.keys())[0], parameter)
        A = LincombOperator([A0, A1, A2, A3], [1, p, p*p, p*p*p])
        B1 = NumpyMatrixOperator(np.array([[1], [0], [1]]))
        B2 = NumpyMatrixOperator(np.array([[0], [1], [0]]))
        B = LincombOperator([B1, B2], [1, p])
        C = NumpyMatrixOperator(np.array([[1, 0, 1]]))
        return LTIModel(A, B, C)
    elif name == 'rational':
        A0 = NumpyMatrixOperator(np.diag([-2, -1, -1]))
        A1 = NumpyMatrixOperator(np.array([[0, -1, 0], [0, 0, -0.5], [0, -0.5, 0]]))
        A2 = NumpyMatrixOperator(np.array([[0.1, 0, 0.2], [0, 1, 0], [-0.2, 0, 0]]))
        A3 = NumpyMatrixOperator(np.array([[0, 1, 0], [-1, 0, 0], [0, -10, 0]]))
        p = ExpressionParameterFunctional(list(parameter.keys())[0], parameter)
        pinv = ExpressionParameterFunctional('1/(p+1)', parameter)
        A = LincombOperator([A0, A1, A2, A3], [1, p, p*p, pinv])
        B1 = NumpyMatrixOperator(np.array([[1], [0], [1]]))
        B2 = NumpyMatrixOperator(np.array([[0], [1], [0]]))
        B = LincombOperator([B1, B2], [1, p])
        C = NumpyMatrixOperator(np.array([[1, 0, 1]]))
        return LTIModel(A, B, C)


def plti_to_tf(sys, r=None, tol=None):
    def tf(p, mu=None):
        H = BlockOperator([[sys.A, sys.B], [sys.C, sys.D]])
        return to_matrix(H.assemble(mu=sys.parameters.parse(p)), format='dense')

    return TransferFunction(dim_input=sys.solution_space.dim+sys.dim_input,
                            dim_output=sys.solution_space.dim+sys.dim_output,
                            tf=tf,
                            dtf=lambda p: p*tf(p))


def estimate_rcond(M, norm=np.inf):
    n = 'I' if np.isinf(norm) else '1'
    return spla.lapack.zgecon(M, spla.norm(M, ord=norm), norm='I' if np.isinf(norm) else '1')[0]


#====================================================================================================
# PLOTTING

ratio = (1 + np.sqrt(5)) / 2  # golden ratio
w = 3.2
double_column = (w, w/ratio)
dpi = 200
font_size = 10.0
cmap = plt.cm.plasma
plt.rcParams.update({
    'figure.figsize': double_column,
    'figure.dpi': dpi,
    'figure.constrained_layout.use': True,
    'lines.linewidth': 1,
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'legend.fontsize': 'medium',
    'pgf.rcfonts': False,
    'image.cmap': cmap.name,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
})
adaptive_opts = {'initial_num': 20}

export_path = Path(__file__).parent / 'figures'


def plot_2derr(sys_orig, sys_itpl, xlim, ylim, rom=None, tol=1e16, path=Path('figures'), kind='', title=r'\(\delta(\omega,p)\)', vmin=1e-16, vmax=1e-2):
    fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(w)
    err = sys_orig - sys_itpl
    s = np.geomspace(*xlim, 127)
    p = np.linspace(*ylim, 127)
    mags, conds = [], []
    if rom is not None:
        Es = rom.B.matrix[:, :-sys_orig.dim_input] @ rom.C.matrix[:-sys_orig.dim_output]
    tic = perf_counter()
    for pi in p:
        if rom is not None:
            K = pi*rom.E.matrix - rom.A.matrix
        try:
            val = err.bode(s, mu=pi)[0] / np.abs(sys_orig.bode(s, mu=pi)[0])
        except ValueError:
            val = np.zeros((len(s), sys_orig.dim_output, sys_orig.dim_input))
        mags.append(np.abs(val, out=np.finfo(np.float64).tiny*np.ones_like(val), where=val!=0))
        if rom is not None:
            conds.append([estimate_rcond(K-1/si*Es) for si in s])
    print(f'{kind}:\t{perf_counter()-tic:.2f}s')
    mags = np.squeeze(np.concatenate(mags, axis=1)).T
    im = ax.pcolormesh(s, p, mags, norm='log', vmin=vmin, vmax=vmax, rasterized=True)
    if len(conds):
        conds = np.array(conds)
        conds = np.power(conds, -1, out=np.inf*np.ones_like(conds), where=conds!=0)
        color = cmap(np.linspace(0,1,20))[-1]
        ax.contour(s, p, conds, levels=[tol], colors=[color], linewidths=0.5)
        plt.legend([Line2D([0], [0], color=color, lw=0.5)], [rf'\(\tilde{{\kappa}}\!=\!10^{{{int(np.log10(tol))}}}\)'], loc='upper left', fontsize='x-small', framealpha=.8)
    ax.set(xscale='log', xlabel=r'\(\omega\)', ylabel=r'\(p\)', xlim=xlim, ylim=ylim)
    cb = plt.colorbar(im, ax=ax, pad=-0.03, shrink=0.819)
    cb.ax.tick_params(labelsize='x-small')
    ax.set_box_aspect(1/ratio)
    path = export_path / path
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'{kind}_2derr')
