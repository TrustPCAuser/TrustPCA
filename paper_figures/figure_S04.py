import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse
import plotly.express as px
import pandas as pd
from tueplots import axes, bundles
import matplotlib.patches as mpatches

from matplotlib.legend_handler import HandlerPatch

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

outfile = 'paper_figures/figure_S04.pdf'

covs = np.load('data/uncertainty_prediction/results/ancient_corrected_stats.npy')

results = pd.read_csv('data/uncertainty_prediction/results/ancient_samples.csv', sep=',', header=0, index_col=0, converters={'in ellipse frequencies': pd.eval})

rates = [0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
rates_p = [20, 50, 75, 90, 95, 99]

confs = [0.25, 0.50, 0.75, 0.95]
palette = px.colors.qualitative.Vivid + px.colors.qualitative.Vivid
palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

pixel = 1/plt.rcParams['figure.dpi']
c1 = mpatches.Circle((0, 0), 1, fc='None', lw=1, edgecolor=palette[6])
c2 = mpatches.Circle((0, 0), 1, fc='None', lw=1, edgecolor=palette[7])

with plt.rc_context({**bundles.aistats2022(family="serif"), **axes.lines()}):
  fig, axes = plt.subplots(2, 3, figsize=(488*pixel, 0.85*488*pixel))
  for j, ax in enumerate(axes.ravel()):
    results_sub = results.loc[results['missing rate']==rates[j]]
    pc1 = results_sub['discr 1'].values
    pc2 = results_sub['discr 2'].values
    pcs = np.array([pc1, pc2])
    true_cov = np.cov(pcs)
    true_mean = np.mean(pcs, axis=1)

    ax.set_title(rf'$r={{{rates_p[j]}}}\,\%$')
    ax.scatter(pc1, pc2, c=palette[7], s=1, alpha=0.3)
    ax.scatter(0, 0, c=palette[6], s=1)
    for conf in confs: 
      h, e = get_ellipse(true_mean, true_cov, conf, palette[7])
      ax.add_patch(e)
      _, e = get_ellipse([0, 0], covs[j], conf, palette[6])
      ax.add_patch(e)

  fig.supxlabel(r'$\tau_1 - \hat{\tau}_1$')
  fig.supylabel(r'$\tau_2 - \hat{\tau}_2$')
  fig.legend([c1, c2], [r'pred.', r'true'], handler_map={mpatches.Circle:HandlerPatch(patch_func=make_legend_ellipse),
                        }, loc='center left', bbox_to_anchor=(1, 0.5), title=r'Contours')
  plt.savefig(outfile)