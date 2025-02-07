import numpy as np
import os 
from utils import *

def PMPs_drift(z, P, genomeans): #single PMP
    """
    This function projects a (ancient) genotype sample (with missing data) to the model plane computed from high-coverage modern individuals.
    This approach results in projections that are similar to projections obtained by SmartPCA.

    Args:
        z (numpy.ndarray): (Ancient) genotype sample (with missing genotypes encoded as "9").
        P (pandas.DataFrame): Model plane built from first two leading eigenvectors.
        genomeans (pandas.DataFrame): Mean genotype value across modern population for all loci.

    Returns:
        numpy.ndarray: coordinates of projected sample.
    """
    # get index of missing genotypes
    index = np.where(z == 9)
 
    # remove missing genotype positions from genotype sample
    z_stern=np.delete(z, index) 
    
    # ... from genomeans vector
    genomeans_stern = np.delete(genomeans, index) 

    # ... from model plane
    P_stern = np.delete(P, index, axis =0)
    
    # center and normalize genotype sample
    snp_drift = np.sqrt((genomeans_stern / 2) * (1 - genomeans_stern/2))
    z_stern_meaned = (z_stern-genomeans_stern) / snp_drift 
     
    # project genotype sample to model plane
    dotp = np.dot(P_stern.T, P_stern).astype(float)
    dotz = np.dot(P_stern.T, z_stern_meaned).astype(float)
    tau = np.dot(np.linalg.inv(dotp), dotz)
    return tau

def transition(ref, alt):
    """
    This function checks if two alleles (reference and alternative) are a transition.

    Args:
        ref (str): Reference allele.
        alt (str): Alternative allele.

    Returns:
        Boolean: transition (True or False).
    """
    return (ref == 'A' and alt == 'G') or (ref == 'G' and alt == 'A') or (ref == 'C' and alt == 'T') or (ref == 'T' and alt == 'C')

def parse_geno(input_data):
    """
    This function parses a .geno file and converts it into a numpy.ndarray where
    rows are individuals and columns are genotypes

    Args:
        input_data (str or file object): File path or file object

    Returns:
        numpy.ndarray: Genotype array
    """
    try:
        if isinstance(input_data, str):
            with open(input_data, 'r') as file:
                geno_lines = file.readlines()
        else:
            geno_lines = input_data.readlines()
    except Exception as e:
        raise ValueError("Input must be a file path or a file object.") from e

    geno_data = [list(map(int, line.strip())) for line in geno_lines]
    geno_array = np.array(geno_data, dtype=np.uint8).T
    return geno_array 

def downsample(ind, P, ds_rate, n, folder):
    """
    This function simulates n downsampled versions of one high-coverage individual with a defined downsampling rate 
    and projects the downsampled versions into a predefined model plane (similar to SmartPCA).

    Args:
        ind (numpy.ndarray): Genotypes for one individual.
        P (pandas.Dataframe): Contains SNP name, genomean and PC axes (eigenvecotors) computed from modern individuals
        ds_rate (float): Amount of SNPs to be randomly removed from a sample, e.g. ds_rate=20 removes 20% of the SNPs.
        n (int): Number of downsamplings per genotype sample.
        folder (str): File path where to save the coordinates of the SmartPCA projections
    """ 
    # evaluate number of missing genotypes to be randomly introduced
    non_nine_indices = np.where(ind != 9)[0]    
    num_new_nines = int(len(non_nine_indices) * ds_rate / 100.0)

    # define output file
    tau_file = open(f"{folder}/{ds_rate}taus.txt", "w")
    print(ds_rate)
    for j in range(n):  # repeat n times
        if (j%1000 == 0):
            print(str(j)+" times downsampled and projected for "+ str(ds_rate))
        ds_ind = np.array(ind, copy=True)
        
        # Randomly select indices to set to 9
        new_nine_indices = np.random.choice(non_nine_indices, num_new_nines, replace=False)
        ds_ind[new_nine_indices] = 9

        # Project to model plane and save projection tau
        tau = PMPs_drift(ds_ind, P[["PC1", "PC2"]], P["genomean"])
        tau_file.write(f"{tau[0]} {tau[1]}\n")
    tau_file.close()

def ds_individuum(ind_name, ind_geno, ds_rates, n, P, output_folder):
    """
    Computes the true SmartPCA projection for a high-coverage (ancient) sample
    and subsequently starts the downsampling simulation.

    Args:
        ind_name (pandas.Dataframe): Metadata of individual
        ind_geno (numpy.ndarray): Genotypes for one individual.
        P (pandas.Dataframe): Contains SNP name, genomean and PC axes (eigenvecotors) computed from modern individuals
        ds_rate (float): Amount of SNPs to be randomly removed from a sample, e.g. ds_rate=20 removes 20% of the SNPs.
        n (int): Number of downsamplings per genotype sample.
        output_folder (str): File path where to save the simulations
    """ 
    # Create output folder specific for individual
    ind_output_folder = output_folder+str(ind_name["Group_ID"])
    os.makedirs(ind_output_folder, exist_ok=True)
    print("created folder")

    # Compute and save true embedding of high-coverage individual similar to SmartPCA
    tau_orig = PMPs_drift(ind_geno, P[["PC1", "PC2"]], P["genomean"])
    print("calc tau orig")
    orig_file=open(str(ind_output_folder)+"/original_tau.txt", "w")
    orig_file.write(str(tau_orig[0])+ " "+str(tau_orig[1]))
    orig_file.close()

    # Start the simulation experiment for each downsampling rate
    print("start downsampling")
    for ds_rate in ds_rates:
        downsample(ind_geno, P, ds_rate, n, ind_output_folder)
    
def process_ancients(geno, names, P, output_folder, indices, n=20000, ds_rates=[20,50,75,90,95,99]):
    """
    Reads in the .geno file and starts the simulation experiment for each individual

    Args:
        geno (numpy.ndarray): Genotypes of individuals (rows: individuals; columns: genotypes)
        names (pandas.Dataframe): Metadata of individuals
        P (pandas.Dataframe): Contains SNP name, genomean and PC axes (eigenvecotors) computed from modern individuals
        ds_rates (list of float): Amount of SNPs to be randomly removed from a sample, e.g. ds_rate=20 removes 20% of the SNPs.
        n (int): Number of downsamplings per genotype sample.
        output_folder (str): File path where to save the simulations
        n (int): Number of downsamplings per genotype sample.
        ds_rates (list of float): Amount of SNPs to be randomly removed from a sample, e.g. ds_rate=20 removes 20% of the SNPs.
    """ 
    # parse geno file
    geno_file_input = parse_geno(geno)

    # Select variant genotypes
    geno_file = select_snps(geno_file_input)

    print("Number of anicent individuals for downsampling: ", len(geno_file))
    
    t=0
    for line in geno_file: # loop through individuals
        name = names.iloc[t]
        print("processing ", name["Group_ID"])
        ds_individuum(name, np.array(line), ds_rates, n,P, output_folder)
        t+=1