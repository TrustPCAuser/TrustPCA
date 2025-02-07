import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
from tueplots import axes, bundles

def read_coordinates(file_path, rate=1):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coordinates = [list(map(float, line.strip().split())) for idx, line in enumerate(lines) if idx % rate == 0]
    return coordinates

def flatten_comprehension(matrix):
  return [item for row in matrix for item in row]

colors = {
        '99taus.txt': 'orange',
        '95taus.txt': 'purple',
        '90taus.txt': 'red',
        '75taus.txt': 'green',
        '50taus.txt': 'blue',
        '20taus.txt': 'gold',
        'original_tau.txt': 'black'
    }


collected_dirs = []
collected_taus = []
originals = []
names_map = {
    '99taus.txt': '99 %',
    '95taus.txt': '95 %',
    '90taus.txt': '90 %',
    '75taus.txt': '75 %',
    '50taus.txt': '50 %',
    '20taus.txt': '20 %',
}

samples_dict = {
    '99 %': [],
    '95 %': [],
    '90 %': [],
    '75 %': [],
    '50 %': [],
    '20 %': [],
}

for root, dirs, files in os.walk("data/downsampling/results"):
  for dir in dirs:
    if dir in ['Altai_Neanderthal.DG', 'Denisova.DG']:
      print(dir)
      folder_path = os.path.join(root, dir)
      collected_dirs.append(dir)
      for j, file in enumerate(colors.keys()):
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
          if file == 'original_tau.txt':
             originals.append(read_coordinates(file_path))
          else:
            collected_taus.append(file)
            coordinates = np.array(read_coordinates(file_path))
            samples_dict[names_map[file]].append(coordinates)

outfile = 'paper_figures/figure_S05.pdf'

originals = np.squeeze(np.array(originals), axis=1)
print(originals)

PC1 = []
PC2 = []
samples = []
rates = []

for key in ['20 %', '50 %', '75 %', '90 %', '95 %', '99 %']:
  samples.extend(flatten_comprehension([np.repeat(i, 20000) for i in range(2)]))
  PC1.extend(np.vstack(samples_dict[key])[:, 0])
  PC2.extend(np.vstack(samples_dict[key])[:, 1])
  rates.extend(flatten_comprehension([np.repeat(key, 20000) for i in range(2)]))

print(len(samples), len(PC1), len(PC2), len(rates))

samples = pd.DataFrame({'PC1': PC1, 
                        'PC2': PC2,
                        'ancients': [str(i+1) for i in samples],
                        'r': rates})

modern_df = pd.read_csv('data/embedding_modern_refs.csv')
rates = [20, 50, 75, 90, 95, 99]
rates_p = ['20 %', '50 %', '75 %', '90 %', '95 %', '99 %']
palette = px.colors.qualitative.Vivid + px.colors.qualitative.Vivid
palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

px = 1/plt.rcParams['figure.dpi']

labels = ['Neanderthal', 'Denisovan']
with plt.rc_context({**bundles.aistats2022(family="serif"), **axes.lines()}):
  fig, axes = plt.subplots(2, 3, figsize=(488*px, 0.75*488*px), sharex=True, sharey=True)
  for j, ax in enumerate(axes.ravel()):
    ax.scatter(modern_df['PC1'], modern_df['PC2'], alpha=0.2, s=1, c='grey')
    ax.set_title(rf'$r={{{rates[j]}}}\,\%$')
    samples_sub = samples.loc[samples['r']==rates_p[j]]
    for i in range(1, 3):
      samples_ancient = samples_sub.loc[samples_sub['ancients']==str(i)]
      ax.scatter(samples_ancient['PC1'], samples_ancient['PC2'], alpha=0.8, s=2, c=palette[i+5], label=labels[i-1], rasterized=True)
    ax.scatter(*originals.T, s=1, c='black')  
  plt.setp(axes[-1, :], xlabel='PC 1')
  plt.setp(axes[:, 0], ylabel='PC 2')  
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title=r'$\hat{\tau}^{(i, j)}$ for $i=$')
  plt.savefig(outfile, dpi=150)