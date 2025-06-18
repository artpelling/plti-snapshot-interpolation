#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import perf_counter

import numpy as np
import scipy.linalg as spla
from pathlib import Path

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockOperator
from pymor.operators.constructions import InverseOperator, LincombOperator, LowRankOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameters
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.bt import BTReductor


#====================================================================================================
# INITIALIZATION

def load_example(name):
    parameter = Parameters({'p': 1})  # create a one dimensional parameter
    if name == 'toy':
        """Toy example from https://doi.org/10.1109/CDC45484.2021.9682841 Sec. V-B"""
        A0 = NumpyMatrixOperator(np.diag([-2, -1, -1]))
        A1 = NumpyMatrixOperator(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
        p = ExpressionParameterFunctional(list(parameter.keys())[0], parameter)
        A = LincombOperator([A0, A1], [1, p])
        B = NumpyMatrixOperator(np.array([[1], [0], [1]]))
        C = NumpyMatrixOperator(np.array([[1, 0, 1]]))
        return LTIModel(A, B, C, name='Toy example system')
    elif name == 'penzl':
        """Parametric Penzl model from https://epubs.siam.org/doi/10.1137/130914619"""
        n = 1000
        A0 = NumpyMatrixOperator(np.array([[-1, 100], [-100, -1]]))
        A1 = NumpyMatrixOperator(np.array([[0, 1], [-1, 0]]))
        p = ExpressionParameterFunctional(list(parameter.keys())[0], parameter)
        A = BlockDiagonalOperator([LincombOperator([A0, A1], [1, p]),
                                   NumpyMatrixOperator(np.array([[-1, 200], [-200, 1]])),
                                   NumpyMatrixOperator(np.array([[-1, 400], [-400, 1]])),
                                   NumpyMatrixOperator(-1 * np.diag(np.arange(n) + 1))])
        B = BlockColumnOperator([NumpyMatrixOperator(10 * np.ones((2, 1))),
                                 NumpyMatrixOperator(10 * np.ones((2, 1))),
                                 NumpyMatrixOperator(10 * np.ones((2, 1))),
                                 NumpyMatrixOperator(np.ones((n, 1)))])
        return LTIModel(A, B, B.H, name='Penzl example system')
    elif name == 'polynomial':
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


def plti_to_tf(sys, r=None, tol=None):
    def tf(p, mu=None):
        if r is not None or tol is not None:
            rom = BTReductor(sys, mu=p).reduce(r=r, tol=tol)
            H = BlockOperator([[rom.A, rom.B], [rom.C, rom.D]])
            return to_matrix(H.assemble(), format='dense')
        else:
            H = BlockOperator([[sys.A, sys.B], [sys.C, sys.D]])
            if p == 100 and False:
                T = np.eye(4)
                T[1:3, 1:3] = np.array([[0, 1], [1, 0]])
                print(T)
                return T @ to_matrix(H.assemble(mu=sys.parameters.parse(p)), format='dense') @ T
            else:
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
w = 3.32153
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
        val = err.bode(s, mu=pi)[0] / np.abs(sys_orig.bode(s, mu=pi)[0])
        mags.append(np.abs(val, out=np.finfo(np.float64).tiny*np.ones_like(val), where=val!=0))
        conds.append([estimate_rcond(K-1/si*Es) for si in s])
    print(f'{kind}:\t{perf_counter()-tic:.2f}s')
    mags = np.squeeze(np.concatenate(mags, axis=1)).T
    im = ax.pcolormesh(s, p, mags, norm='log', vmin=vmin, vmax=vmax, rasterized=True)
    if len(conds):
        conds = np.array(conds)
        conds = np.power(conds, -1, out=np.inf*np.ones_like(conds), where=conds!=0)
        contour = ax.contour(s, p, conds, levels=[tol], colors=[cmap(np.linspace(0,1,20))[-1]], linewidths=0.5)
        ax.clabel(contour, inline=1, fontsize='x-small', fmt=rf'\(\tilde{{\kappa}}\!=\!10^{{{int(np.log10(tol))}}}\)', inline_spacing=25)
    ax.set(title=title, xscale='log', xlabel=r'\(\omega\)', ylabel=r'\(p\)', xlim=xlim, ylim=ylim)
    cb = plt.colorbar(im, ax=ax, pad=-0.03, shrink=0.885)
    cb.ax.tick_params(labelsize='x-small')
    ax.set_box_aspect(1/ratio)
    path = export_path / path
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'{kind}_2derr')
