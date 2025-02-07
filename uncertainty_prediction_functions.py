import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def var_discrepency(V_obs, var_tau_r):
  """
  Computes the variance discrepancy between the estimated embedding and the true embedding.

  Parameters:
  - V_obs (numpy.ndarray): Eigenvector matrix at the observed features.
  - var_tau_r (numpy.ndarray): The expected covariance matrix of the PCs (all but the first two).

  Returns:
  - numpy.ndarray: The variance in discrepancy between the estimated embedding and the true embedding.
  """
  matrix_of_linear_map = - np.linalg.inv(V_obs[:, 0:2].T @ V_obs[:, 0:2]) @ V_obs[:, 0:2].T @ V_obs[:, 2:]
  return matrix_of_linear_map @ var_tau_r @ matrix_of_linear_map.T

def get_ellipse(mean, Sigma, confidence_level, color):
  """
  Computes an 2-dimensional ellipse as defined by the mean and covariancematrix and confidence lebel of a Gaussian distribution.

  Parameters:
  - mean (numpy.ndarray): Mean of Gaussian.
  - Sigma (numpy.ndarray): Covariance matrix of Gaussian.
  - confidence_level (float): Confidence level.
  - color (str): Color of ellipse

  Returns:
  - matplotlib.patches.Patch: The Ellipse.
  """
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

  ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fc='None', lw=1, edgecolor=color)
  return ellipse