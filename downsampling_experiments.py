### Downsampling simulation experiments ###
# simulation for one sample and different downsampling rates #
# repeat for all samples of interest #

import pandas as pd 

from downsampling_functions import *

# Model plane P --> PC space computed from modern population
P = pd.read_csv('data/downsampling/P.csv', header=0)

# Genotype indices that are considered (540,247 loci)
indices = pd.read_csv('data/SNPs_mwe.csv', header=0)

output_folder = "data/downsampling/results/"

######################################
########## Experiment 1 ##############
######################################

# High-coverage ancient samples used for simulation (17 samples including Neanderthal and Denisovan)
ancients_geno = "data/downsampling/ancients_ds.geno"
ancients_list = pd.read_csv("data/downsampling/ancients_ds_ids.csv", header=1, skiprows=0, sep=" ")

process_ancients(ancients_geno, ancients_list, P, output_folder, indices)