import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tueplots import axes, bundles

import plotly.express as px

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import matplotlib as mpl


def make_legend_ellipse(legend, orig_handle,
                        xdescent, ydescent,
                        width, height, fontsize):
    p = mpatches.Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                         width = width+xdescent, height=(height+ydescent))

    return p

def get_ellipse(mean, Sigma, confidence_level, color):
  chi2_val = chi2.ppf(confidence_level, df=2)
  eigvals, eigvecs = np.linalg.eigh(Sigma)

  # Sorting eigenvalues and corresponding eigenvectors
  order = eigvals.argsort()[::-1]
  eigvals = eigvals[order]
  eigvecs = eigvecs[:, order]

  # Width and height of the ellipse (2 * sqrt(eigenvalue * chi-square value))
  width = 2 * np.sqrt(eigvals[0] * chi2_val)
  height = 2 * np.sqrt(eigvals[1] * chi2_val)

  # Angle of the ellipse in degrees (in the direction of the largest eigenvector)
  angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

  ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fc='None', lw=1, edgecolor=color, label='confidence ellipse (0.95)')
  return height/2, ellipse

covs_uc = np.load('data/uncertainty_prediction/results/ancient_uncorrected_stats.npy')
covs_c = np.load('data/uncertainty_prediction/results/ancient_corrected_stats.npy')
results = pd.read_csv('data/uncertainty_prediction/results/ancient_samples.csv', sep=',', header=0, index_col=0, converters={'in ellipse frequencies': pd.eval})

rates = [0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
rates_p = [20, 50, 75, 90, 95, 99]
j = 2

confs = [0.25, 0.50, 0.75, 0.95]

results_sub = results.loc[results['missing rate']==rates[j]]
pc1 = results_sub['discr 1'].values
pc2 = results_sub['discr 2'].values
pcs = np.array([pc1, pc2])
print(pcs.shape)
true_cov = np.cov(pcs)
true_mean = np.mean(pcs, axis=1)

pixel = 1/plt.rcParams['figure.dpi']
palette = px.colors.qualitative.Vivid + px.colors.qualitative.Vivid
palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]
c1 = mpatches.Circle((0, 0), 1, fc='None', lw=1, edgecolor=palette[6])
c2 = mpatches.Circle((0, 0), 1, fc='None', lw=1, edgecolor=palette[7])
c3 = Line2D([], [], marker='.', color=palette[7], linestyle='None', alpha=0.6)

var_a_norm = np.load('paper_figures/variance_estimation/var_a_nn.npy')
var_m_norm = np.load('paper_figures/variance_estimation/var_m_nn.npy')
factors = np.load('data/factors.npy')

with plt.rc_context({**bundles.aistats2022(family="serif"), **axes.lines()}):
  mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
  fig, axs = plt.subplot_mosaic([['a)', 'b)', 'c)'], ['a)', 'b)', 'd)']], figsize=(488*pixel, 0.5*488*pixel), )
  #fig, axs = plt.subplots(1, 3, figsize=(488*pixel, 0.5*488*pixel), )
  #plt.hist(np.clip(var_a_nn, a_min=0, a_max=10), 50)
  axs['a)'].hist(var_a_norm, 50, label=r'ancient', color=palette[10], alpha=0.6)
  axs['a)'].hist(var_m_norm, 50, label=r'modern', color='black', alpha=0.6)
  axs['a)'].set_xlabel(r'Locus variance')
  axs['a)'].legend()
  axs['b)'].hist(factors, 50, color=palette[10], alpha=0.6)
  axs['b)'].set_xlabel(r'PC variance factor $f^\text{out}$')
  axs['b)'].axvline(x=np.mean(factors), color='black', alpha=0.6, linestyle='--')
  axs['b)'].text(x=1, y=410, s=r'$\mu_{f^\text{out}} = 1.86$', horizontalalignment='left', fontsize=7, c='black', alpha=0.6)
  #axs[2].set_title(rf'$r={{{rates_p[j]}}}\,\%$')
  axs['c)'].scatter(pc1, pc2, c=palette[7], s=1, alpha=0.3)
  axs['c)'].scatter(0, 0, c=palette[6], s=1)
  axs['d)'].scatter(pc1, pc2, c=palette[7], s=1, alpha=0.3)
  axs['d)'].scatter(0, 0, c=palette[6], s=1)
  for conf in confs: 
    h, e = get_ellipse(true_mean, true_cov, conf, palette[7])
    axs['c)'].add_patch(e)
    _, e = get_ellipse([0, 0], covs_uc[j], conf, palette[6])
    axs['c)'].add_patch(e)
    h, e = get_ellipse(true_mean, true_cov, conf, palette[7])
    axs['d)'].add_patch(e)
    _, e = get_ellipse([0, 0], covs_c[j], conf, palette[6])
    axs['d)'].add_patch(e)
  axs['d)'].sharex(axs['c)'])
  axs['c)'].tick_params(labelbottom=False)
  axs['d)'].set_xlabel(r'$\tau_1 - \hat{\tau}_1$')
  axs['d)'].set_ylabel(r'$\tau_2 - \hat{\tau}_2$')
  axs['c)'].set_ylabel(r'$\tau_2 - \hat{\tau}_2$')

  axs['a)'].text(
        0.0, 1.0, r'\textbf{a)}', transform=(
            axs['a)'].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), va='bottom')
  axs['b)'].text(
        0.0, 1.0, r'\textbf{b)}', transform=(
            axs['b)'].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), va='bottom')
  axs['c)'].text(
        0.0, 1.0, r'\textbf{c)}', transform=(
            axs['c)'].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), va='bottom')
  axs['d)'].text(
        0.0, 1.0, r'\textbf{d)}', transform=(
            axs['d)'].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), va='bottom')
  axs['c)'].set_title(r'no correction')
  axs['d)'].set_title(r'with correction')
  axs['d)'].legend([c1, c2], [r'pred.', r'empir.'], handler_map={mpatches.Circle:HandlerPatch(patch_func=make_legend_ellipse),
                        }, loc='upper center', bbox_to_anchor=(0.5, 1.85), ncol=3)
  plt.savefig('paper_figures/figure_03.pdf', dpi=150)