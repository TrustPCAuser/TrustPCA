'''
Experiments to evaluate the accuracy and performance of the uncertainty prediction framework.
Given high-coverage (ancient) individuals, we evaluate the empirical spread of these samples around their true embedding when genotypes are randomly removed. 
This spread in terms of a covariance matrix is then compared to the covariance matrix that is predicted by the proposed prediction framework.
'''
import numpy as np
import pandas as pd
from scipy.stats import chi2
from utils import *
from uncertainty_prediction_functions import *


def is_point_in_ellipse(sample, mean, Sigma, confidence_levels):
  """
  Checks if a sample lies within a specified confidence ellipse given the Gaussian distribution N(mean, Sigma).

  Parameters:
  - sample (numpy.ndarray): test sample
  - mean (numpy.ndarray): mean
  - Sigma (numpy.ndarray): covariance matrix
  - confidence_levels (list of floats): confidence levels

  Returns:
  - list of Booleans: whether sample lies within (True) or outside of (False) the confidence ellipse.
  """
  # Calculate the chi-squared threshold value for the given confidence level

  chi2_vals = [chi2.ppf(i, df=2) for i in confidence_levels]  # For 2D

  # Compute the Mahalanobis distance of the sample from the mean
  diff = sample - mean
  mahalanobis_distance_squared = np.dot(np.dot(diff.T, np.linalg.inv(Sigma)), diff)
    
  # Check if the Mahalanobis distance is within the chi-squared threshold
  return [mahalanobis_distance_squared <= i for i in chi2_vals]

def simulate_sample(size, rate):
  """
  Simulates observed/missing indices for an array of size "size".

  Parameters:
  - size (int): total size of positions
  - rate (float): missing rate

  Returns:
  - numpy.ndarray, numpy.ndarray: sorted arrays of observed and missing indices
  """
  observed = int(np.round(size * (1-rate)))
  all_indices = np.arange(0, size, 1)

  observed_indices = np.random.choice(all_indices, size=(1, observed), replace=False) # missing_rate = False rate
  missing_indices = np.delete(all_indices, observed_indices)
  return np.sort(observed_indices[0]), np.sort(missing_indices)


# specify parameters ------------------------------------------------------------------------------------------------
n_samples = 1 # remove n_sample sets of genotypes at given rate
missing_rates = [0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------
"""

Modern samples

"""
# -------------------------------------------------------------------------------------------------------------------

# Load data
# Modern genotype data --> High-coverage modern individuals (all 540247 loci known)
try:
  modern_genotypes = np.load('data/modern_genotypes.npy')
except FileNotFoundError:
  modern_genotypes = load_and_preprocess_data()
  np.save('data/modern_genotypes.npy', modern_genotypes)

# eigenvectors, eigenvalues
try:
  V = np.load('data/eigenvectors.npy')
  Lambda = np.load('data/eigenvalues.npy')
except FileNotFoundError:
  Lambda, V = compute_PCA(modern_genotypes)

# Compute projection (true embedding)
taus = modern_genotypes @ V[:, 0:2]

n_features = modern_genotypes.shape[1]
output = []
vars = []

for missing_rate in missing_rates:
  print(f'missing rate: {missing_rate}')

  # simulate missing positions
  s = [simulate_sample(size=n_features, rate=missing_rate) for i in range(n_samples)]
  observed_inds, missing_inds = zip(*s)

  print('simulated data')

  for i in range(n_samples): # for one simulation, compute SmartPCA projection of all modern genotypes
    # extract only simulated observed loci from high-coverage modern individuals
    modern_obs = modern_genotypes[:, observed_inds[i]]
    
    # eigenvectors at observed positions
    V_obs = V[observed_inds[i], 0:2]

    # SmartPCA projection of downsampled individuals
    proj_factor = np.linalg.inv(V_obs.T @ V_obs) @ V_obs.T
    tau_ests = [proj_factor @ z_obs for z_obs in modern_obs]

    # Compute discrepancy between true projection and the estimates of downsampled samples computed by SmartPCA
    diff = taus - tau_ests
    print('computed tau ests')

    # Predict covariance matrix of the discrepancy
    var_dis = var_discrepency(V[observed_inds[i]], np.diag(Lambda[2:]))
    vars.append(var_dis)

    print('computed var dis')

    # Check if the SmartPCA projection lie within the predicted confidence ellipses for different confidence levels 
    in_ellipse_frequencies = [is_point_in_ellipse(taus[i], tau_ests[i], var_dis, confidence_levels) for i in range(modern_genotypes.shape[0])]

    # format results
    for j in range(modern_genotypes.shape[0]):
      output.append([missing_rate, i, diff[j][0], diff[j][1], in_ellipse_frequencies[j], j])
    
    # save results
    outfile1 = 'data/uncertainty_prediction/results/modern_samples.csv'
    output_df = pd.DataFrame(output, 
                             columns=['missing rate', 'simulation sample', 'discr 1', 'discr 2', 
                                      'in ellipse frequencies', 'genotype sample nr'])
    output_df.to_csv(outfile1)

    outfile2 = 'data/uncertainty_prediction/results/modern_stats.npy'
    np.save(outfile2, np.array(vars))



# -------------------------------------------------------------------------------------------------------------------
"""

Ancient samples

"""
# -------------------------------------------------------------------------------------------------------------------

output = []
vars = []
vars_corr = []

# Load ancient samples
all_ancients = load_all_ancients(replace_nan=False)

# Find and select high-coverage (>90%) ancient individuals
nan_counts = [np.count_nonzero(~np.isnan(all_ancients[i])) for i in range(np.shape(all_ancients)[0])]
high_coverage_ancient_indices = np.where(np.array(nan_counts)/np.shape(all_ancients)[1] > 0.9)[0]

# Load ancient samples again, but replace non-observed genotypes by genomean
all_ancients = load_all_ancients(replace_nan=True)

# Select previously defined set of high-coverage ancients
ancient_genotypes = all_ancients[high_coverage_ancient_indices]

# Load factors required to predict the descrepancy
factors = np.load('data/factors.npy')

# projection (true embedding)
taus = ancient_genotypes @ V[:, 0:2]

for missing_rate in missing_rates: 
  # simulate missing/observed positions
  s = [simulate_sample(size=n_features, rate=missing_rate) for i in range(n_samples)]
  observed_inds, missing_inds = zip(*s)

  print('simulated data')

  for i in range(n_samples): # for one simulation, compute SmartPCA projection of selected ancient genotypes
    # extract only simulated observed loci from high-coverage ancient individuals
    ancient_obs = ancient_genotypes[:, observed_inds[i]]

    # eigenvectors at observed positions
    V_obs = V[observed_inds[i], 0:2]

    # SmartPCA projection of downsampled individuals
    proj_factor = np.linalg.inv(V_obs.T @ V_obs) @ V_obs.T
    tau_ests = [proj_factor @ z_obs for z_obs in ancient_obs]

    # Compute discrepancy between true projection and the estimates of downsampled samples computed by SmartPCA
    diff = taus - tau_ests
    print('computed tau ests')

    # Predict covariance matrix of the discrepancy
    # using correction factor
    var_dis = var_discrepency(V[observed_inds[i]], np.diag(Lambda[2:] * factors[2:]))
    vars_corr.append(var_dis)
    # not using correction factor
    var_dis = var_discrepency(V[observed_inds[i]], np.diag(Lambda[2:]))
    vars.append(var_dis)
    print('computed var dis')

    # Check if the SmartPCA projection lie within the predicted confidence ellipses for different confidence levels 
    in_ellipse_frequencies = [is_point_in_ellipse(taus[i], tau_ests[i], var_dis, confidence_levels) for i in range(ancient_genotypes.shape[0])]
    
    # format results
    for j in range(ancient_genotypes.shape[0]):
      output.append([missing_rate, i, diff[j][0], diff[j][1], in_ellipse_frequencies[j], j])
    
    # save results
    outfile1 = 'data/uncertainty_prediction/results/ancient_samples.csv'
    output_df = pd.DataFrame(output, 
                             columns=['missing rate', 'simulation sample', 'discr 1', 'discr 2', 
                                      'in ellipse frequencies', 'genotype sample nr'])
    output_df.to_csv(outfile1)

    outfile2 = 'data/uncertainty_prediction/results/ancient_uncorrected_stats.npy'
    np.save(outfile2, np.array(vars))
    outfile2 = 'data/uncertainty_prediction/results/ancient_corrected_stats.npy'
    np.save(outfile2, np.array(vars_corr))


# More general experiment with more downsampling samples to quantify the accuracy of the spread
# specify parameters ------------------------------------------------------------------------------------------------
n_samples = 100 # remove n_sample sets of genotypes at given rate
missing_rates = [0.2, 0.5, 0.75, 0.9, 0.95, 0.99]
confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
"""

Modern samples

"""
# -------------------------------------------------------------------------------------------------------------------

# Compute projection (true embedding)
taus = modern_genotypes @ V[:, 0:2]

n_features = modern_genotypes.shape[1]
output = []
vars = []

for missing_rate in missing_rates:
  print(f'missing rate: {missing_rate}')

  # simulate missing positions
  s = [simulate_sample(size=n_features, rate=missing_rate) for i in range(n_samples)]
  observed_inds, missing_inds = zip(*s)

  print('simulated data')

  for i in range(n_samples): # for one simulation, compute SmartPCA projection of all modern genotypes
    # extract only simulated observed loci from high-coverage modern individuals
    modern_obs = modern_genotypes[:, observed_inds[i]]
    
    # eigenvectors at observed positions
    V_obs = V[observed_inds[i], 0:2]

    # SmartPCA projection of downsampled individuals
    proj_factor = np.linalg.inv(V_obs.T @ V_obs) @ V_obs.T
    tau_ests = [proj_factor @ z_obs for z_obs in modern_obs]

    # Compute discrepancy between true projection and the estimates of downsampled samples computed by SmartPCA
    diff = taus - tau_ests
    print('computed tau ests')

    # Predict covariance matrix of the discrepancy
    var_dis = var_discrepency(V[observed_inds[i]], np.diag(Lambda[2:]))
    vars.append(var_dis)

    print('computed var dis')

    for j in range(modern_genotypes.shape[0]):
      output.append([missing_rate, i, diff[j][0], diff[j][1], j])
    

    outfile1 = 'data/uncertainty_prediction/results/modern_samples_for_quantification.csv'
    output_df = pd.DataFrame(output, columns=['missing rate', 'simulation sample', 'discr 1', 'discr 2', 'genotype sample nr'])
    output_df.to_csv(outfile1)

    outfile2 = 'data/uncertainty_prediction/results/modern_stats_for_quantification.npy'
    np.save(outfile2, np.array(vars))

# -------------------------------------------------------------------------------------------------------------------
"""

Ancient samples

"""
# -------------------------------------------------------------------------------------------------------------------
# projection (true embedding)
taus = ancient_genotypes @ V[:, 0:2]

for missing_rate in missing_rates: 
  # simulate missing/observed positions
  s = [simulate_sample(size=n_features, rate=missing_rate) for i in range(n_samples)]
  observed_inds, missing_inds = zip(*s)

  print('simulated data')

  for i in range(n_samples): # for one simulation, compute SmartPCA projection of selected ancient genotypes
    # extract only simulated observed loci from high-coverage ancient individuals
    ancient_obs = ancient_genotypes[:, observed_inds[i]]

    # eigenvectors at observed positions
    V_obs = V[observed_inds[i], 0:2]

    # SmartPCA projection of downsampled individuals
    proj_factor = np.linalg.inv(V_obs.T @ V_obs) @ V_obs.T
    tau_ests = [proj_factor @ z_obs for z_obs in ancient_obs]

    # Compute discrepancy between true projection and the estimates of downsampled samples computed by SmartPCA
    diff = taus - tau_ests
    print('computed tau ests')

    # Predict covariance matrix of the discrepancy
    # using correction factor
    var_dis = var_discrepency(V[observed_inds[i]], np.diag(Lambda[2:] * factors[2:]))
    vars_corr.append(var_dis)
    # not using correction factor
    var_dis = var_discrepency(V[observed_inds[i]], np.diag(Lambda[2:]))
    vars.append(var_dis)
    print('computed var dis')

    for j in range(ancient_genotypes.shape[0]):
      output.append([missing_rate, i, diff[j][0], diff[j][1], j])
    
    outfile1 = 'data/downsampling/results/ancient_samples_for_quantification.csv'
    output_df = pd.DataFrame(output, columns=['missing rate', 'simulation sample', 'discr 1', 'discr 2', 'genotype sample nr'])
    output_df.to_csv(outfile1)

    outfile2 = 'data/uncertainty_prediction/results/ancient_uncorrected_stats_for_quantification.npy'
    np.save(outfile2, np.array(vars))
    outfile2 = 'data/uncertainty_prediction/results/ancient_corrected_stats_for_quantification.npy'
    np.save(outfile2, np.array(vars_corr))