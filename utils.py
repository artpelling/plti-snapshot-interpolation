#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import scipy.linalg as spla

from functools import partial
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


def parametric_lti_to_tf(sys, r=None, tol=None):
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


def assemble_ptf(Ep, A, B, C, X, Y, parameter, tol=1e16):
    Es = Y@X
    def tf(s, mu=None):
        p = mu['p'][0]
        K = p*Ep - A
        rcond = estimate_rcond(K-1/s*Es)
        if rcond == 0 or tol > 1/rcond:
            # K = np.linalg.inv(s*K - Es)
            # return s * C @ K @ B
            K = np.linalg.inv(K-1/s*Es)
            return C @ K @ B
        else:
            K = -np.linalg.inv(K)
            XK = X @ K
            KY = K @ Y
            H = np.linalg.inv(s*np.eye(X.shape[0]) + XK @ Y)
            H = KY @ H @ XK + K
            return C @ H @ B

    return TransferFunction(dim_input=B.shape[1], dim_output=C.shape[0], tf=tf, parameters=parameter)


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


def plot_mags(sys_orig, sys_itpl, P, xlim, name=''):
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=cmap(np.linspace(0, 1, len(P)+2)[1:-1]))
    for mu in P:
        sys_itpl.mag_plot(xlim, mu=mu, ax=ax, dB=False, label=rf'\(p = {int(mu["p"][0])}\)', adaptive_opts=adaptive_opts)
        sys_orig.mag_plot(xlim, mu=mu, ax=ax, dB=False, color='k', linestyle=':', adaptive_opts=adaptive_opts)
    ax.set(title='', xlabel='Frequency [rad/s]', xlim=xlim, ylabel='Magnitude')
    ax.grid()
    ax.legend(fontsize='x-small')
    fig.set_figwidth(w)
    ax.set_box_aspect(1/ratio)
    fig.savefig(export_path / f'{name}_mag')


def plot_spectra(IL, ILs, r, xmax, ylim=None, markevery=1, name=''):
    fig = plt.figure()
    fig, ax = plt.subplots()
    col = cmap(np.linspace(0, 1, 4))
    ax.set_prop_cycle(color=col[:-1][::-1])
    svs = spla.svdvals(np.hstack([IL, ILs]))
    ranks = np.arange(len(svs))+1
    svL = spla.svdvals(IL)
    svLs = spla.svdvals(ILs)
    ax.semilogy(ranks, svL/svL[0], label=r'\(\sigma(\mathbb{L})\)', marker='o', markevery=markevery)
    ax.semilogy(ranks, svLs/svLs[0], label=r'\(\sigma(\mathbb{L}_s)\)', marker='o', markevery=markevery, zorder=3)
    ax.semilogy(ranks, svs/svs[0], label=r'\(\sigma([\mathbb{L}~\mathbb{L}_s])\)', marker='s', markerfacecolor='none',
                markevery=markevery, linestyle='-.', zorder=4)
    if markevery == 1:
        ax.set_xticks(ranks)
    ax.axvline(r, color=col[-1], linestyle='-', label=r'\(r\)', path_effects=[pe.Stroke(linewidth=1.5, foreground='grey'), pe.Normal()], zorder=2)
    ax.set(xlabel='Order', xlim=(1, xmax), ylabel='\(\sigma\)', ylim=ylim, title='Singular values')
    ax.grid()
    ax.legend()
    ax.tick_params(axis='y', which='major', pad=25)
    [l.set_ha('left') for l in ax.get_yticklabels()]
    fig.set_figwidth(w)
    ax.set_box_aspect(1/ratio)
    fig.savefig(export_path / f'{name}_svs')


def plot_2derr(sys_orig, sys_itpl, Ep, A, X, Y, xlim, ylim, tol=1e16, name='', title=r'\(\delta(\omega,p)\)', vmin=1e-18, vmax=1e-8):
    fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(w)
    err = sys_orig - sys_itpl
    s = np.geomspace(*xlim, 100)
    p = np.linspace(*ylim, 100)
    mags, conds = [], []
    Es = Y@X
    from time import perf_counter
    tic = perf_counter()
    for pi in p:
        K = pi*Ep - A
        val = err.bode(s, mu=pi)[0]
        mags.append(np.abs(val, out=np.finfo(np.float64).tiny*np.ones_like(val), where=val!=0))
        conds.append([estimate_rcond(K-1/si*Es) for si in s])
    print(perf_counter()-tic)
    mags = np.squeeze(np.concatenate(mags, axis=1)).T
    conds = np.array(conds)
    conds = np.power(conds, -1, out=np.inf*np.ones_like(conds), where=conds!=0)
    im = ax.pcolormesh(s, p, mags, norm='log', vmin=vmin, vmax=vmax, rasterized=True)
    contour = ax.contour(s, p, conds, levels=[tol], colors=[cmap(np.linspace(0,1,20))[-1]], linewidths=0.5)
    plt.clabel(contour, inline=1, fontsize='x-small', fmt=rf'\(\tilde{{\kappa}}\!=\!10^{{{int(np.log10(tol))}}}\)', inline_spacing=75)
    contour.collections[0].set_label('tol')
    ax.set(title=title, xscale='log', xlabel=r'\(\omega\)', ylabel=r'\(p\)', xlim=xlim, ylim=ylim)
    cb = plt.colorbar(im, ax=ax, pad=-0.03, shrink=0.314)
    cb.ax.tick_params(labelsize='x-small')
    ax.set_box_aspect(1/ratio)
    fig.savefig(export_path / f'{name}_2derr')


def plot_cond(Ep, A, X, Y, xlim, ylim, tol=1e16, name='', title=r'\(\kappa(p\mathcal{E}_p-\mathcal{A}-s^{-1}\mathcal{E}_s)\)', vmin=1e-20, vmax=1e-5):
    fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_figwidth(w)
    s = np.geomspace(*xlim, 100)
    p = np.linspace(*ylim, 100)
    mags = []
    Es = Y@X
    for pi in p:
        K = pi*Ep - A
        mags.append([np.linalg.cond(K-1/si*Es) for si in s])
    mags = np.array(mags)
    im = ax.pcolormesh(s, p, mags, norm='log', rasterized=True)
    ax.contour(s, p, mags, levels=[tol], colors=cmap([1]))
    ax.set(title=title, xscale='log', xlabel=r'\(s\)', ylabel=r'\(p\)', xlim=xlim, ylim=ylim)
    cb = plt.colorbar(im, ax=ax, pad=-0.03, shrink=0.314)
    cb.ax.tick_params(labelsize='x-small')
    ax.set_box_aspect(1/ratio)
    fig.savefig(export_path / f'{name}_cond')
