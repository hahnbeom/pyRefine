import os
import numpy as np

# BLOSUM62 matrix
# In Nao's, use all (24)
# In Minkyung's, use regular aas (20)
# ARNDCQEGHILKMFPSTWYVBZX*
blosum_mtx = np.array([[  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],   
                       [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],   
                       [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],   
                       [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],   
                       [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],   
                       [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],   
                       [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],   
                       [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],   
                       [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],   
                       [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],   
                       [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],   
                       [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],   
                       [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],   
                       [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],   
                       [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],   
                       [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],   
                       [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],   
                       [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],   
                       [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],   
                       [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4]]).astype(np.float32)
blosum_mtx /= 10.0 # normalize (roughly around -1 to 1)

# Meiler features
#residue,steric_parameter,polarizability,volume,hydrophobicity,isoelectric_pt,helix_prob,sheet_prob
meiler_mtx = np.array([[1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23], # ALA
                       [2.34, 0.29, 6.13,-1.01,10.74, 0.36, 0.25], # ARG
                       [1.60, 0.13, 2.95,-0.60, 6.52, 0.21, 0.22], # ASN
                       [1.60, 0.11, 2.78,-0.77, 2.95, 0.25, 0.20], # ASP
                       [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41], # CYS
                       [1.56, 0.18, 3.95,-0.22, 5.65, 0.36, 0.25], # GLN
                       [1.56, 0.15, 3.78,-0.64, 3.09, 0.42, 0.21], # GLU
                       [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15], # GLY
                       [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30], # HIS
                       [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45], # ILE
                       [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31], # LEU
                       [1.89, 0.22, 4.77,-0.99, 9.99, 0.32, 0.27], # LYS
                       [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32], # MET
                       [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38], # PHE
                       [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34], # PRO
                       [1.31, 0.06, 1.60,-0.04, 5.70, 0.20, 0.28], # SER
                       [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36], # THR
                       [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42], # TRP
                       [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41], # TYR
                       [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49]  # VAL
                       ]).astype(np.float32)
meiler_mtx /= 5.0 # normalize

three_to_one = {} 
# ARNDCQEGHILKMFPSTWYVBZX*
three_to_one['ALA'] = 'A'  
three_to_one['ARG'] = 'R'
three_to_one['ASN'] = 'N'
three_to_one['ASP'] = 'D'
three_to_one['CYS'] = 'C'
three_to_one['GLN'] = 'Q'
three_to_one['GLU'] = 'E'
three_to_one['GLY'] = 'G'
three_to_one['HIS'] = 'H'
three_to_one['ILE'] = 'I'
three_to_one['LEU'] = 'L'
three_to_one['LYS'] = 'K'
three_to_one['MET'] = 'M'
three_to_one['PHE'] = 'F'
three_to_one['PRO'] = 'P'
three_to_one['SER'] = 'S'
three_to_one['THR'] = 'T'
three_to_one['TRP'] = 'W'
three_to_one['TYR'] = 'Y'
three_to_one['VAL'] = 'V'

#atom_types = ['CAbb', 'CH3', 'CObb', 'Nbb', 'OCbb', 'CH2', 'CH1', 'Npro'] # index
rosetta_atom_types = np.load("%s/atypes.npy"%os.path.dirname(__file__)) # contains atom type indices
                                                            # 0th idx: residues ARNDCQEGHILKMFPSTWYV
                                                            # 1th idx: atoms N, CA, C, O, CB

def blosum(seq):
    return blosum_mtx[seq]

def meiler(seq):
    return meiler_mtx[seq]

def reweight_seq(msa1hot, cutoff=0.8):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = np.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
    id_mask = id_mtx > id_min
    w = 1.0 / np.sum(id_mask, axis=-1)
    return w

def msa2pssm(msa1hot, w):
    beff = np.sum(w)
    f_i = np.sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = np.sum(-f_i * np.log(f_i+1e-9), axis=1)
    return np.concatenate((f_i, h_i[:,None]), axis=-1)

def seq2pssm(msa1hot):
    f_i = np.sum(msa1hot, axis=0) + 1e-9
    h_i = np.sum(-f_i * np.log(f_i+1e-9), axis=-1)
    return np.concatenate((f_i, h_i[:,None]), axis=-1)
