import numpy as np
import pandas as pd

database_path = '/local_scratch/ancientPCA/ancientPCA/database/'
output_db = '/local_scratch/TrustPCA/database/'

V = np.load(database_path + 'eigenvectors.npy')
np.save(output_db + 'eigenvectors.npy', V[:, 0:2])

evs = np.load(database_path + 'eigenvalues.npy')
np.save(output_db + 'eigenvectors.npy', evs[0:2])

d = pd.read_csv(database_path + 'coordinates_MWE.csv')
d[['Group', 'Class', 'PC1', 'PC2']].to_csv(output_db + 'coordinates_MWE.csv')
