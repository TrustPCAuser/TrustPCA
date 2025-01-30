import numpy as np
import pandas as pd

database_path = '/local_scratch/ancientPCA/ancientPCA/database/'
output_db = '/local_scratch/TrustPCA/database/'

V = np.load(database_path + 'eigenvectors.npy')
V = np.array(V, dtype=np.float32)
#np.savez_compressed(output_db + 'eigenvectors2', V=V)
np.save(output_db + 'eigenvectors.npy', V)


#evs = np.load(database_path + 'eigenvalues.npy')
#np.save(output_db + 'eigenvalues.npy', evs)

#d = pd.read_csv(database_path + 'coordinates_MWE.csv')
#d[['Group', 'Class', 'PC1', 'PC2']].to_csv(output_db + 'coordinates_MWE.csv')

#d = pd.read_csv(database_path + 'genomean.csv')
#print(np.array(d['x'].values).shape)