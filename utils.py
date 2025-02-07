import numpy as np
import pandas as pd

def select_snps(x):
  """
  Filters the matrix of genotypes for variant loci.

  Args:
      x (numpy.ndarray): Matrix of genotypes
  
  Returns:
      numpy.ndarray: Filtered matrix of genotypes
  """
  if x[1]==540247:
        return x
  else:
    present_modern_snps_path = 'data/SNPs_mwe.csv'
    present_modern_snps = pd.read_csv(present_modern_snps_path, header=0)
    modern_snps_indices = present_modern_snps['x'].values
    return np.array([i[modern_snps_indices-1] for i in x])

def parse_geno_file_path(file_path):
  """
  This function parses a .geno file and converts it into a numpy.ndarray where
  rows are individuals and columns are genotypes

  Args:
      input_data (str): File path

  Returns:
      numpy.ndarray: Genotype array
  """
  with open(file_path, 'r') as file:
      geno_lines = file.readlines()
  
  geno_data = []
  for idx, line in enumerate(geno_lines):
      #if idx % 100000 == 0 and idx > 0:
      #    print(f"Parsed {idx} lines of 540248")
      geno_data.append(list(map(int, line.strip())))
  
  #print("transposing")
  d = np.array(geno_data, dtype=np.uint8).T
  return(np.where(d==9., np.nan, d))

def load_and_preprocess_data(replace_nan=True):
  """
  Loads all modern genotypes, selects the relevant snps, centers and normalizes, 
  and optionally replaces non-observed genotypes by the genomean.

  Args:
    replace_nan (Boolean): Whether or not to replace non-observed genotypes by the genomean.
  
  Returns:
    numpy.ndarray: Matrix of modern genotypes
  """
  x = parse_geno_file_path('data/modern_genotypes.geno')
  
  x_s = select_snps(x)

  # normalization
  genomean_path = 'data/genomean.csv'
  genomean = pd.read_csv(genomean_path, header=0)
  genomean = genomean['x'].values
  snp_drift = np.sqrt((genomean / 2) * (1 - genomean/2))

  if replace_nan:
    # replace nans by genomean
    x_s_i = np.nan_to_num(x_s, nan=genomean)
  else:
    x_s_i = x_s

  # normalization
  x_s_i_c = (x_s_i - genomean) / snp_drift

  return x_s_i_c

def load_all_ancients(replace_nan=True):
  """
  Loads all ancient genotypes, centers and normalizes, 
  and optionally replaces non-observed genotypes by the genomean.

  Args:
    replace_nan (Boolean): Whether or not to replace non-observed genotypes by the genomean.
  
  Returns:
    numpy.ndarray: Matrix of ancient genotypes
  """
  x = np.load('data/genotype_ancient_refs.npy')
  
  # normalization
  genomean_path = 'data/genomean.csv'
  genomean = pd.read_csv(genomean_path, header=0)
  genomean = genomean['x'].values
  snp_drift = np.sqrt((genomean / 2) * (1 - genomean/2))

  # replace nans by genomean
  if replace_nan:
    x_s_i = np.nan_to_num(x, nan=genomean)
  else:
    x_s_i = x

  # normalization
  x_s_i_c = (x_s_i - genomean) / snp_drift

  return x_s_i_c

def compute_PCA(x):
  """
  Computes singular value decomposition of a matrix x and computes eigenvalues from singular values
  
  Args:
      x (numpy.ndarray): Matrix
  
  Returns:
      numpy.ndarray, numpy.ndarray: Positive (and non-zero) eigenvalues and eigenvectors of X^T*X
  """
  _, Sigma, V = np.linalg.svd(x, full_matrices=False)

  # Flip first eigenvector to obtain commonly known orientation of PC plot
  V[0] = -1*V[0]

  # Compute eigenvalues from singular values
  Lambda = Sigma**2/(x.shape[0] - 1)

  return Lambda, V.T