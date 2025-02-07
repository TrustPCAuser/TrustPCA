import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import pandas as pd
from tueplots import axes, bundles
from matplotlib.patches import Patch
import matplotlib.text as mtext

palette = px.colors.qualitative.Vivid + px.colors.qualitative.Vivid

def kl_divergence(m0, m1, C0, C1):
   '''
   N0(m0, C0): true distribution
   N1(m1, C1): predicted distribution
   returns KL(N0||N1)'''
   C1_inv = np.linalg.inv(C1)
   return 0.5 * (np.trace(C1_inv @ C0) - 2 + (m1 - m0).T @ C1_inv @ (m1 - m0) + np.log(np.linalg.det(C1)/np.linalg.det(C0)))

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title
    
outfile = 'paper_figures/figure_S02.pdf'

covs_modern = np.load('data/uncertainty_prediction/results/modern_stats_for_quantification.npy')
covs_uncorrected = np.load('data/uncertainty_prediction/results/ancient_uncorrected_stats_for_quantification.npy')
covs_corrected = np.load('data/uncertainty_prediction/results/ancient_corrected_stats_for_quantification.npy')

results_modern = pd.read_csv('data/uncertainty_prediction/results/modern_samples_for_quantification.csv', sep=',', header=0, index_col=0, converters={'in ellipse frequencies': pd.eval})
results_ancient = pd.read_csv('data/uncertainty_prediction/results/ancient_samples_for_quantification.csv', sep=',', header=0, index_col=0, converters={'in ellipse frequencies': pd.eval})

rates = [0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
rates_p = [20, 50, 75, 90, 95, 99]

def compute_kl_divergences(rates, results, covs):
  kl_per_rate_mean = []
  kl_per_rate_std = []
  for j, rate in enumerate(rates):
    results_m = results.loc[results['missing rate']==rates[j]]
    kl_per_sample = []
    for i in range(100):
      results_sub = results_m.loc[results['simulation sample']==i]
      pc1 = results_sub['discr 1'].values
      pc2 = results_sub['discr 2'].values
      pcs = np.array([pc1, pc2])
      true_cov = np.cov(pcs)
      #print('true', true_cov)
      pred_cov = covs[j*100+i]
      true_mean = np.mean(pcs, axis=1)
      pred_mean = np.array([0, 0])
      kl_div = kl_divergence(true_mean, pred_mean, true_cov, pred_cov)
      kl_per_sample.append(kl_div)
    kl_per_rate_mean.append(np.mean(kl_per_sample))
    kl_per_rate_std.append(np.std(kl_per_sample))
  return(kl_per_rate_mean, kl_per_rate_std)


palette = px.colors.qualitative.Vivid + px.colors.qualitative.Vivid
palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

pixel = 1/plt.rcParams['figure.dpi']

with plt.rc_context({**bundles.aistats2022(family="serif"), **axes.lines()}):
  plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsmath}'
  })
  fig, ax = plt.subplots(1, 1, figsize=(488*pixel, 0.68*488*pixel))
  kl_per_rate_mean, kl_per_rate_std = compute_kl_divergences(rates, results_modern, covs_modern)
  ax.scatter(rates, kl_per_rate_mean, label='Modern', c=palette[6], s=10)
  ax.errorbar(x=rates,y=kl_per_rate_mean, yerr=kl_per_rate_std, fmt='none', capsize=3, c=palette[6])
  kl_per_rate_mean, kl_per_rate_std = compute_kl_divergences(rates, results_ancient, covs_uncorrected)
  ax.scatter(rates, kl_per_rate_mean, label='Ancient no corr.', c=palette[7], s=10)
  ax.errorbar(x=rates,y=kl_per_rate_mean, yerr=kl_per_rate_std, fmt='none', capsize=3, c=palette[7])
  kl_per_rate_mean, kl_per_rate_std = compute_kl_divergences(rates, results_ancient, covs_corrected)
  ax.scatter(rates, kl_per_rate_mean, label='Ancient with corr.', c=palette[7], s=10, marker='s')
  ax.errorbar(x=rates,y=kl_per_rate_mean, yerr=kl_per_rate_std, fmt='none', capsize=3, c=palette[7])
  ax.set_xlabel(r'$r$')
  ax.set_ylabel(r'$D_\text{KL}(\mathcal{N}_\text{empir.}||\mathcal{N}_\text{pred.})$')
  plt.legend(['Used data', (Patch(color=palette[6])), (Patch(color=palette[7])), 'Correction', (plt.Line2D([], [], linestyle='', marker='.', color='black')), (plt.Line2D([], [], linestyle='', marker='s', markersize=3, color='black'))], ['', 'Modern', 'Ancient', '', 'False', 'True'],
           handler_map={str: LegendTitle({'fontsize': 8})})
  fig.savefig('paper_figures/figure_S02.pdf')