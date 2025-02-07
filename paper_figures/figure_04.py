import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
#import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
#import plotly.express as px
from tueplots import axes, bundles
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from matplotlib.lines import Line2D
import itertools

palette = px.colors.qualitative.Vivid
palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

pixel = 1/plt.rcParams['figure.dpi']

def gaussian_density_within_ellipse(mean, Sigma, confidence_level):
    # Step 1: Compute chi-square value and eigen decomposition
    chi2_val = chi2.ppf(confidence_level, df=2)
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    # Sorting eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Width and height of the ellipse (2 * sqrt(eigenvalue * chi-square value))
    width = 2 * np.sqrt(eigvals[0] * chi2_val)
    height = 2 * np.sqrt(eigvals[1] * chi2_val)

    # Get the bounds for the meshgrid
    x_min, x_max = mean[0] - width / 2, mean[0] + width / 2
    y_min, y_max = mean[1] - height / 2, mean[1] + height / 2

    # Create the meshgrid
    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Step 2: Compute the Gaussian density
    rv = multivariate_normal(mean=mean, cov=Sigma)
    pdf_values = rv.pdf(points)
    pdf_values = pdf_values.reshape(xx.shape)

    # Step 3: Compute Mahalanobis distance and mask outside points
    points_centered = points - mean
    Sigma_inv = np.linalg.inv(Sigma)
    mahalanobis_distances = np.einsum('ij,jk,ik->i', points_centered, Sigma_inv, points_centered)
    mahalanobis_distances = mahalanobis_distances.reshape(xx.shape)

    # Mask points outside the 95% confidence ellipse
    pdf_values[mahalanobis_distances > chi2_val] = np.nan
    return xx, yy, pdf_values

def make_legend_ellipse(legend, orig_handle,
                        xdescent, ydescent,
                        width, height, fontsize):
    p = mpatches.Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                         width = width+xdescent, height=(height+ydescent))

    return p

modern_df = pd.read_csv('data/embedding_modern_refs.csv')
groups = pd.read_csv('paper_figures/real_world/modern_groups_curated.csv', header=0)
vars = np.load('paper_figures/real_world/variance_ancients_high_uncertainty.npy')
s_emb = np.load('paper_figures/real_world/mean_ancients_high_uncertainty.npy')
metadata_ancients = pd.read_csv('paper_figures/real_world/metadata_ancients_high_uncertainty.csv', header=0)

vars_low = np.load('paper_figures/real_world/variance_ancients_low_uncertainty.npy')
s_emb_low = np.load('paper_figures/real_world/mean_ancients_low_uncertainty.npy')
metadata_ancients_low = pd.read_csv('paper_figures/real_world/metadata_ancients_low_uncertainty.csv', header=0)

modern_df['Group'] = groups['Group']
modern_df = modern_df.sort_values(by=['Group'])

c1 = mpatches.Circle((0, 0), 1, fc='grey', lw=0.5, ec='lightgrey')
c2 = mpatches.Circle((0, 0), 1, fc='None', lw=0.5, edgecolor='red')

markers = Line2D.filled_markers
markers = ['o', '^', 's', 'p', '*', 'D', 'd', 'P', 'X']
style = list(itertools.product(palette, markers))

modern_df['Style'] = modern_df['Group'].map(dict(zip(modern_df['Group'].unique(),
                                       style)))

with plt.rc_context({**bundles.aistats2022(family="serif"), **axes.lines()}):
  fig, ax = plt.subplots(1, 1, figsize=(488*pixel, 0.75*488*pixel))
  for sty in modern_df['Style'].unique():
    df_sub = modern_df.loc[modern_df['Style']==sty]
    ax.scatter(df_sub['PC1'], df_sub['PC2'], alpha=1, s=4, marker=sty[1], c=sty[0], label=df_sub['Group'].unique()[0], edgecolor='white', linewidth=0.1)
  for i, emb in enumerate(s_emb_low):  
    ax.text(emb[0]+4, emb[1]-2, str(i+5), fontsize=7, color='black')
  for i, emb in enumerate(s_emb):  
    ax.text(emb[0]-15, emb[1]-18, str(i+1), fontsize=7, color='black')
  for i in range(len(vars)):
    mean = s_emb[i]
    cov = vars[i]
    xx, yy, pdf_values = gaussian_density_within_ellipse(mean, cov, 0.95)
    ax.contourf(xx, yy, pdf_values, levels=15, cmap='Greys', alpha=0.7, antialiased=False)
  for i in range(len(vars_low)):
    mean = s_emb_low[i]
    cov = vars_low[i]
    xx, yy, pdf_values = gaussian_density_within_ellipse(mean, cov, 0.95)
    ax.contourf(xx, yy, pdf_values, levels=10, cmap='Greys', alpha=0.7, antialiased=False)
  ax.scatter(*s_emb.T, s=2, c='black', alpha=0.2)
  ax.set_xlabel(r'PC 1')
  ax.set_ylabel(r'PC 2')
  ax.set_ylim(-380, 170)
  fig.legend(ncols=4, markerscale=1.3, title=r'Modern West Eurasians', alignment='left', borderpad=0.2, labelspacing=0.3, handlelength=0.5, handletextpad=0.3, columnspacing=1.2, fontsize=5, title_fontsize=6, loc='lower left', bbox_to_anchor=(0.12, 0.12))
  legend2 = plt.legend([c1], [r'Ancients'], alignment='left', handler_map={mpatches.Circle:HandlerPatch(patch_func=make_legend_ellipse)}, borderpad=0.4, labelspacing=0.3, handletextpad=0.3, fontsize=5, title_fontsize=6)

  fig.savefig('paper_figures/figure_04.pdf', dpi=150)

