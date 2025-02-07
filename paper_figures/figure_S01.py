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
    '99.5taus.txt': '99.5 %',
    '99taus.txt': '99 %',
    '98taus.txt': '98 %',
    '97taus.txt': '97 %',
    '96taus.txt': '96 %',
    '95taus.txt': '95 %',
    '94taus.txt': '94 %',
    '93taus.txt': '93 %',
    '92taus.txt': '92 %',
    '90taus.txt': '90 %',
    '75taus.txt': '75 %',
    '50taus.txt': '50 %',
    '20taus.txt': '20 %',
    'original_tau.txt': '0 %'
}

samples_dict = {
    '99.5 %': [],
    '99 %': [],
    '98 %': [],
    '97 %': [],
    '96 %': [],
    '95 %': [],
    '94 %': [],
    '93 %': [],
    '92 %': [],
    '90 %': [],
    '75 %': [],
    '50 %': [],
    '20 %': [],
}

for root, dirs, files in os.walk("/data/downsampling/results"):
  for dir in dirs:
    if dir not in ['Altai_Neanderthal.DG', 'Denisova.DG']:
      print(dir)
      folder_path = os.path.join(root, dir)
      collected_dirs.append(dir)
      for j, file in enumerate(names_map.keys()):
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
          if file == 'original_tau.txt':
             originals.append(read_coordinates(file_path))
          else:
            collected_taus.append(file)
            coordinates = np.array(read_coordinates(file_path))
            print('file', file)
            samples_dict[names_map[file]].append(coordinates)

outfile = 'paper_figures/figure_S01.pdf'

originals = np.squeeze(np.array(originals), axis=1)
print(originals)

PC1 = []
PC2 = []
samples = []
rates = []

for key in ['20 %', '50 %', '75 %', '90 %', '92 %', '93 %', '94 %', '95 %','96 %','97 %','98 %', '99 %', '99.5 %']:
  print(key)
  samples.extend(flatten_comprehension([np.repeat(i, 20000) for i in range(15)]))
  PC1.extend(np.vstack(samples_dict[key])[:, 0])
  PC2.extend(np.vstack(samples_dict[key])[:, 1])
  rates.extend(flatten_comprehension([np.repeat(key, 20000) for i in range(15)]))

print(len(samples), len(PC1), len(PC2), len(rates))

samples = pd.DataFrame({'PC1': PC1, 
                        'PC2': PC2,
                        'ancients': [str(i+1) for i in samples],
                        'r': rates})

ref_pc1 = [originals[int(i)-1, 0] for i in samples['ancients']]
ref_pc2 = [originals[int(i)-1, 1] for i in samples['ancients']]

samples["id"] = samples.index
samples['discrepancyPC1'] = ref_pc1 - samples['PC1']
samples['discrepancyPC2'] = ref_pc2 - samples['PC2']

palette = px.colors.qualitative.Vivid + px.colors.qualitative.Vivid
palette = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in palette]

# Now we group by 'r' and create separate violins for each group
r_values = sorted(samples['r'].unique())  # Get unique values of 'r'

modern_df = pd.read_csv('data/embedding_modern_refs.csv')

px = 1/plt.rcParams['figure.dpi']

with plt.rc_context({**bundles.aistats2022(family="serif"), **axes.lines()}):
  # Create figure and axes
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(488*px, 0.5*488*px))

  # Define positions for each violin group, spaced apart
  positions = np.arange(1, len(r_values) + 1)  # X-positions for violins
  flierprops = dict(marker='.', markersize=2,
                  )
  boxprops = dict(linewidth=1)
  medianprops = dict(linewidth=1)
  ax1.axvline(0, color='grey', linewidth=0.1)
  # Plot the violins for each 'r' value
  for idx, r_val in enumerate(r_values):
    # Subset the data for the current 'r'
    data_PC1 = samples.loc[samples['r'] == r_val, 'discrepancyPC1']
    parts_A = ax1.violinplot(data_PC1, vert=False, positions=[positions[idx]], showmeans=False, showmedians=False, showextrema=False, widths=0.4)
    # Customize appearance (colors, transparency, edge color)
    for pc in parts_A['bodies']:
      pc.set_facecolor(palette[6])  # Blue for PC1
      pc.set_edgecolor('black')
      pc.set_alpha(1)
  ax1.set_xlabel(r'$\tau_1^{(i)} - \hat{\tau}_1^{(i, j)}$')
  ax1_twin = ax1.twinx()

  ax1_twin.scatter(modern_df['PC1'], modern_df['PC2'], alpha=0.1, s=1, c='grey')
  ax1_twin.axes.get_xaxis().set_visible(False)
  ax1_twin.axes.get_yaxis().set_visible(False)

  rates = [20, 50, 75, 90, 92, 93, 94, 95, 96, 97, 98, 99, 99.5]
  # Customize plot aesthetics
  ax1.set_yticks(positions)
  ax1.set_yticklabels([rf'${{{rate}}}$' for rate in rates])
  ax1.set_ylabel(r'$r$ in $\%$')

  ax2.axhline(0, color='grey', linewidth=0.1)
  for idx, r_val in enumerate(r_values):
    # Subset the data for the current 'r'
    data_PC2 = samples.loc[samples['r'] == r_val, 'discrepancyPC2']
    parts_B = ax2.violinplot(data_PC2, vert=True, positions=[positions[idx]], showmeans=False, showmedians=False, showextrema=False, widths=0.4)
  # Customize appearance (colors, transparency, edge color)
    for pc in parts_B['bodies']:
      pc.set_facecolor(palette[6])  # Blue for PC1
      pc.set_edgecolor('black')
      pc.set_alpha(1)
  ax2.set_ylabel(r'$\tau_2^{(i)} - \hat{\tau}_2^{(i, j)}$')
  ax2_twin = ax2.twiny()
  ax2_twin.scatter(modern_df['PC1'], modern_df['PC2'], alpha=0.1, s=1, c='grey')
  ax2_twin.axes.get_xaxis().set_visible(False)
  ax2_twin.axes.get_yaxis().set_visible(False)
  
  rates = [20, 50, 75, 90, 92, 93, 94, 95, 96, 97, 98, 99, 99.5]
  # Customize plot aesthetics
  ax2.set_xticks(positions, [rf'${{{rate}}}$' for rate in rates], rotation=45, ha='right')
  ax2.set_xlabel(r'$r$ in $\%$')

  plt.savefig(outfile)