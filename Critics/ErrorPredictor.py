# Init from master
#from pyrosetta import *
#init("-constant_seed -mute all")

from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import math
import multiprocessing
import scipy
import scipy.spatial
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))+'/DeepAccNet'

import sys
sys.path.insert(0, SCRIPTPATH)
sys.path.insert(0, '%s/utils/'%SCRIPTPATH)
sys.path.insert(0, '%s/src/'%SCRIPTPATH)
from dataProcessingUtils import *
import pyprotein as pyp
from resnet import *
from model import *

def parse_opts(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Error predictor network", epilog="v0.0.1")
    parser.add_argument("infolder", action="store", help="input folder name full of pdbs")
    parser.add_argument("outfolder", action="store", help="output folder name")
    parser.add_argument("--multiDecoy", "-mm", action="store_true", default=False, help="running multi-multi model option (Default: False)")
    parser.add_argument("--noEnsemble", "-ne", action="store_true", default=False, help="running without model ensembling (Default: False)")
    parser.add_argument("--leavetemp", "-lt", action="store_true", default=False, help="leaving temporary files (Default: False)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="verbose flag (Default: False)")
    parser.add_argument("--process", "-p", action="store", type=int, default=1, help="# of cpus to use for featurization (Default: 1)")
    
    if len(argv) == 1:
        parser.print_help()
        return
    
    params = parser.parse_args(argv)
    return params 

def extract_multi_distance_map(pose):
    # Get CB to CB distance map use CA if CB does not exist
    x1 = pyp.get_distmaps(pose, atom1="CB", atom2="CB", default="CA")
    # Get CA to CA distance map
    x2 = pyp.get_distmaps(pose, atom1=pyp.dict_3LAA_to_tip, atom2=pyp.dict_3LAA_to_tip)
    # Get Tip to Tip distancemap
    x3 = pyp.get_distmaps(pose, atom1="CA", atom2=pyp.dict_3LAA_to_tip)
    # Get Tip to CA distancemap
    x4 = pyp.get_distmaps(pose, atom1=pyp.dict_3LAA_to_tip, atom2="CA")
    output = np.stack([x1,x2,x3,x4], axis=-1)
    return output

# Given a scored pose object with N amino acids, 
# returns a tensor of N by N by (# of 2 body interactions)+3
def extract_EnergyDistM(pose, energy_terms):

    # Get the number of residues in the protein.
    length = int(pose.total_residue())
    
    # Prepare distance matrix
    tensor = np.zeros((1+len(energy_terms)+2, length, length))
    
    # Obtain energy graph
    energies = pose.energies()
    graph = energies.energy_graph()
    
    ######################################
    # Fill the dist_matrix with energies #
    ######################################
    aas = []
    for i in range(length):
        index1 = i + 1
        aas.append(pose.residue(index1).name().split(":")[0].split("_")[0])
        
        # Get an edge iterator
        iru = graph.get_node(index1).const_edge_list_begin()
        irue = graph.get_node(index1).const_edge_list_end()
        
        # Parse the energy graph.
        while iru!=irue:
            # Dereference the pointer and get the other end.
            edge = iru.__mul__()
            
            # Evaluate energy edge and get energy values
            evals = [edge[e] for e in energy_terms]
            index2 = edge.get_other_ind(index1)
            
            count = 1
            for k in range(len(evals)):
                e = evals[k]
                t = energy_terms[k]
                
                # For hbond_bb_sc and hbond_sc, just note the presence
                if t == pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc or t == pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc:
                    if e != 0.0:
                        tensor[count, index1-1, index2-1] = 1
                # Otherwise record the original values.
                else:
                    tensor[count, index1-1, index2-1] = e
                    
                count += 1
            # Move pointer
            iru.plus_plus()
    
    #########################################
    # Simple transformation of energy terms #
    #########################################
    for i in range(1, 1+len(evals)):
        temp = tensor[i]
        if i == 1 or i == 2:
            tensor[i] = np.arcsinh(np.abs(temp))/3.0
        elif i == 3 or i==4 or i==5:
            tensor[i] = np.tanh(temp)
            
    #############################################
    # Use CB idstance (CA if CB does not exist) #
    # to calculate distance between residues    #
    #############################################
    for i in range(length):
        for j in range(length):
            index1 = i + 1    
            index2 = j + 1
            # Calculate distance and store. Use CA if CB does not exist.
            if pose.residue(index1).has("CB"):
                vector1 = pose.residue(index1).xyz("CB")
            else:
                vector1 = pose.residue(index1).xyz("CA")
            if pose.residue(index2).has("CB"):
                vector2 = pose.residue(index2).xyz("CB")
            else:
                vector2 = pose.residue(index2).xyz("CA")
            distance = vector1.distance(vector2)
            
            tensor[0, index1-1, index2-1] = distance #1-sigmoid(displacement, scale=10, offset=-5)
    
    ##################################
    # Fill in the hbonds information #
    ##################################
    hbonds = get_hbonds(pose)
    for hb in hbonds[0]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1-1, index2-1] = 1
    count +=1
    for hb in hbonds[1]:
        index1 = hb[0]
        index2 = hb[1]
        tensor[count, index1-1, index2-1] = 1
        
    return tensor, aas

#####################################################
# Given a scored pose object with N amino acids,    #
# returns a tensor of N by N matrix of CB distances #
#####################################################
def extract_distM(pose):
    
    # Get the number of residues in the protein.
    length = int(pose.total_residue())
    
    # Prepare distance matrix
    tensor = np.zeros((1, length, length))
    
    # Obtain energy graph
    energies = pose.energies()
    graph = energies.energy_graph()
    
    # Fill the dist_matrix
    aas = []
    for i in range(length):
        index1 = i + 1
        aas.append(pose.residue(index1).name().split(":")[0].split("_")[0])
        
        # Get an edge iterator
        iru = graph.get_node(index1).const_edge_list_begin()
        irue = graph.get_node(index1).const_edge_list_end()
        
        # Parse the energy graph.
        while iru!=irue:
            # Dereference the pointer and get the other end.
            edge = iru.__mul__()
            index2 = edge.get_other_ind(index1)

            # Calculate distance and store. Use CA if CB does not exist.
            if pose.residue(index1).has("CB"):
                vector1 = pose.residue(index1).xyz("CB")
            else:
                vector1 = pose.residue(index1).xyz("CA")
            if pose.residue(index2).has("CB"):
                vector2 = pose.residue(index2).xyz("CB")
            else:
                vector2 = pose.residue(index2).xyz("CA")
            distance = vector1.distance(vector2)
            
            # Only store information in tensor if residues are less than 15 angs away.
                
            tensor[0, index1-1, index2-1] = distance 
            # Move pointer
            iru.plus_plus()
            
    return tensor, aas

#####################################################
# Given a scored pose object with N amino acids,    #
# returns a tensor of N by N matrix of CB distances #
#####################################################
def extract_AAs_properties_ver1(aas):
    _prop = np.zeros((20+24+1+7, len(aas)))
    for i in range(len(aas)):
        aa = aas[i]
        _prop[residuemap[aa], i] = 1
        _prop[20:44, i] = blosummap[aanamemap[aa]]
        _prop[44, i] = min(i, len(aas)-i)*1.0/len(aas)*2
        _prop[45:, i] = meiler_features[aa]/5
    return _prop

def get_coords(p):

    nres = pyrosetta.rosetta.core.pose.nres_protein(p)

    # three anchor atoms to build local reference frame
    N = np.stack([np.array(p.residue(i).atom('N').xyz()) for i in range(1,nres+1)])
    Ca = np.stack([np.array(p.residue(i).atom('CA').xyz()) for i in range(1,nres+1)])
    C = np.stack([np.array(p.residue(i).atom('C').xyz()) for i in range(1,nres+1)])

    # recreate Cb given N,Ca,C
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466

    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = ca * a + cb * b + cc * c

    return N, Ca, C, Ca+Cb


def set_lframe(pdict):

    # local frame
    z = pdict['Cb'] - pdict['Ca']
    z /= np.linalg.norm(z, axis=-1)[:,None]

    x = np.cross(pdict['Ca']-pdict['N'], z)
    x /= np.linalg.norm(x, axis=-1)[:,None]

    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1)[:,None]

    xyz = np.stack([x,y,z])

    pdict['lfr'] = np.transpose(xyz, [1,0,2])


def get_dihedrals(a, b, c, d):

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

def get_angles(a, b, c):
    
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]
    
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]
    
    x = np.sum(v*w, axis=1)

    return np.arccos(x)

def set_neighbors6D(pdict):

    N = pdict['N']
    Ca = pdict['Ca']
    Cb = pdict['Cb']
    nres = pdict['nres']
    
    dmax = 20.0
    
    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    
    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    
    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
    
    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    
    pdict['dist6d'] = dist6d
    pdict['omega6d'] = omega6d
    pdict['theta6d'] = theta6d
    pdict['phi6d'] = phi6d


def set_neighbors3D(pdict):

    # get coordinates of all non-hydrogen atoms
    # and their types
    xyz = []
    types = []
    pose = pdict['pose']
    nres = pdict['nres']
    for i in range(1,nres+1):
        r = pose.residue(i)
        rname = r.name()[:3]
        for j in range(1,r.natoms()+1):
            aname = r.atom_name(j).strip()
            name = rname+'_'+aname
            if not r.atom_is_hydrogen(j) and aname != 'NV' and aname != 'OXT' and name in atypes:
                xyz.append(r.atom(j).xyz())
                types.append(atypes[name])

    xyz = np.array(xyz)
    xyz_ca = pdict['Ca']
    lfr = pdict['lfr']

    # find neighbors and project onto
    # local reference frames
    dist = 14.0
    kd = scipy.spatial.cKDTree(xyz)
    kd_ca = scipy.spatial.cKDTree(xyz_ca)
    indices = kd_ca.query_ball_tree(kd, dist)
    idx = np.array([[i,j,types[j]] for i in range(len(indices)) for j in indices[i]])

    xyz_shift = xyz[idx.T[1]] - xyz_ca[idx.T[0]]
    xyz_new = np.sum(lfr[idx.T[0]] * xyz_shift[:,None,:], axis=-1)

    #
    # discretize
    #
    nbins = 24
    width = 19.2

    # total number of neighbors
    N = idx.shape[0]

    # bin size
    h = width / (nbins-1)
    
    # shift all contacts to the center of the box
    # and scale the coordinates by h
    xyz = (xyz_new + 0.5 * width) / h

    # residue indices
    i = idx[:,0].astype(dtype=np.int16).reshape((N,1))
    
    # atom types
    t = idx[:,2].astype(dtype=np.int16).reshape((N,1))
    
    # discretized x,y,z coordinates
    klm = np.floor(xyz).astype(dtype=np.int16)

    # atom coordinates in the cell it occupies
    d = xyz - np.floor(xyz)

    # trilinear interpolation
    klm0 = np.array(klm[:,0]).reshape((N,1))
    klm1 = np.array(klm[:,1]).reshape((N,1))
    klm2 = np.array(klm[:,2]).reshape((N,1))
    
    V000 = np.array(d[:,0] * d[:,1] * d[:,2]).reshape((N,1))
    V100 = np.array((1-d[:,0]) * d[:,1] * d[:,2]).reshape((N,1))
    V010 = np.array(d[:,0] * (1-d[:,1]) * d[:,2]).reshape((N,1))
    V110 = np.array((1-d[:,0]) * (1-d[:,1]) * d[:,2]).reshape((N,1))

    V001 = np.array(d[:,0] * d[:,1] * (1-d[:,2])).reshape((N,1))
    V101 = np.array((1-d[:,0]) * d[:,1] * (1-d[:,2])).reshape((N,1))
    V011 = np.array(d[:,0] * (1-d[:,1]) * (1-d[:,2])).reshape((N,1))
    V111 = np.array((1-d[:,0]) * (1-d[:,1]) * (1-d[:,2])).reshape((N,1))

    a000 = np.hstack([i, klm0, klm1, klm2, t, V111])
    a100 = np.hstack([i, klm0+1, klm1, klm2, t, V011])
    a010 = np.hstack([i, klm0, klm1+1, klm2, t, V101])
    a110 = np.hstack([i, klm0+1, klm1+1, klm2, t, V001])

    a001 = np.hstack([i, klm0, klm1, klm2+1, t, V110])
    a101 = np.hstack([i, klm0+1, klm1, klm2+1, t, V010])
    a011 = np.hstack([i, klm0, klm1+1, klm2+1, t, V100])
    a111 = np.hstack([i, klm0+1, klm1+1, klm2+1, t, V000])

    a = np.vstack([a000, a100, a010, a110, a001, a101, a011, a111])
    
    # make sure projected contacts fit into the box
    b = a[(np.min(a[:,1:4],axis=-1) >= 0) & (np.max(a[:,1:4],axis=-1) < nbins) & (a[:,5]>1e-5)]
    
    pdict['idx'] = b[:,:5].astype(np.uint16)
    pdict['val'] = b[:,5].astype(np.float16)


def set_features1D(pdict):

    p = pdict['pose']
    nres = pdict['nres']
    
    # beta-strand pairings
    DSSP = pyrosetta.rosetta.core.scoring.dssp.Dssp(p)
    bbpairs = np.zeros((nres, nres)).astype(np.uint8)
    for i in range(1,nres+1):
        for j in range(i+1,nres+1):
            # parallel
            if DSSP.paired(i,j,0):
                bbpairs[i,j] = 1
                bbpairs[j,i] = 1
            # anti-parallel
            elif DSSP.paired(i,j,1):
                bbpairs[i,j] = 2
                bbpairs[j,i] = 2
    
    abc = np.array(list("BEGHIST "), dtype='|S1').view(np.uint8)
    dssp8 = np.array(list(DSSP.get_dssp_unreduced_secstruct()),
                     dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        dssp8[dssp8 == abc[i]] = i
    dssp8[dssp8 > 7] = 7

    # 3-state DSSP to integers ∈ [0..2]
    DSSP = pyrosetta.rosetta.core.scoring.dssp.Dssp(p)
    abc = np.array(list("EHL"), dtype='|S1').view(np.uint8)
    dssp3 = np.array(list(DSSP.get_dssp_secstruct()), 
                     dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        dssp3[dssp3 == abc[i]] = i
    dssp3[dssp3 > 2] = 2

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    seq = np.array(list(p.sequence()), dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq[seq == alphabet[i]] = i
    
    # backbone (phi,psi)
    phi = np.array(np.deg2rad([p.phi(i) for i in range(1, nres+1)])).astype(np.float32)
    psi = np.array(np.deg2rad([p.psi(i) for i in range(1, nres+1)])).astype(np.float32)

    # termini & linear chainbreaks
    mask1d = np.ones(nres).astype(np.bool)
    mask1d[0] = mask1d[-1] = 0
    for i in range(1,nres):
        A = p.residue(i).atom('CA')
        B = p.residue(i+1).atom('CA')
        if (A.xyz() - B.xyz()).norm() > 4.0:
            mask1d[i-1] = 0
            mask1d[i] = 0

    pdict['seq'] = seq
    pdict['dssp8'] = dssp8
    pdict['dssp3'] = dssp3
    pdict['phi'] = phi
    pdict['psi'] = psi
    pdict['mask1d'] = mask1d
    pdict['bbpairs'] = bbpairs


def init_pose(pose):
    
    pdict = {}
    
    # load PDB file
    pdict['pose'] = pose
    pdict['nres'] = pyrosetta.rosetta.core.pose.nres_protein(pdict['pose'])


    # set coords & local frames
    pdict['N'], pdict['Ca'], pdict['C'], pdict['Cb'] = get_coords(pdict['pose'])
    set_lframe(pdict)

    # set neighbors
    set_neighbors6D(pdict)
    set_neighbors3D(pdict)

    # other features
    set_features1D(pdict)

    return pdict

# function definitions
def get_energy_string(pose, res_pos, scorefxn):
    # given a pose, residue position, and score function
    # returns an energy_string
    scorefxn(pose)
    energy_obj = pose.energies()
    res_energies = energy_obj.residue_total_energies(res_pos)
    energy_string = str(res_energies)
    return energy_string

def energy_string_to_dict(energy_string):
    # given an energy_string
    # returns a dictionary (string --> float) of ALL energy terms
    energy_string = energy_string.replace(") (", ")\n(")
    energy_string = energy_string.replace("( ", "").replace(")", "")
    energy_list = energy_string.split("\n")
    energy_dict = {}
    for element in energy_list:
        (score_term, val) = element.split("; ")
        energy_dict[score_term] = float(val)
    return energy_dict

def remove_nonzero_scores(energy_dict):
    # given an energy_dict
    # returns an energy_dict with trivial scores removed
    result = {}
    for score_term in energy_dict:
        if energy_dict[score_term] != 0:
            result[score_term] = energy_dict[score_term]
    return result

def get_energy_string_quick(energy_obj, res_pos):
    # given an energy_obj and a residue position
    # returns an energy_string
    res_energies = energy_obj.residue_total_energies(res_pos)
    energy_string = str(res_energies)
    return energy_string

def get_one_body_score_terms(pose, scorefxn, score_terms):
    # GIVEN: a pose, a score function, and a list of score terms
    # note that score_terms is a list of strings. these strings must be
    # names of score terms spelled as in the energy_string.
    # RETURNS: one_body_score_terms as a 2d numpy array
    # the rows are residues
    # and the columns are the score terms.
    one_body_score_terms = [] # a list of lists
    scorefxn(pose)
    energy_obj = pose.energies()
    for pos in range(1, len(pose.sequence()) + 1):
        energy_string = get_energy_string_quick(energy_obj, pos)
        energy_dict = energy_string_to_dict(energy_string)
        res_scores = []
        for term in score_terms:
            res_scores.append(energy_dict[term])
        one_body_score_terms.append(res_scores)
    return np.array(one_body_score_terms).T

def mydot(v1, v2):
    result = 0
    for ele in range(3):
        result = result + v1[ele] * v2[ele]
    return result

def angle_between_vecs(v1, v2):
    return math.acos(v1.dot(v2) / (v1.norm() * v2.norm()))

def get_bond_lengths_and_angles(mypose,k):
    # GIVEN: a pose and a residue number k
    # RETURNS: a dictionary where keys are features
    # and values are the values of those features

    # backbone bond lenths and angles
    # for residue k:
    # three bonds to consider:
    # N(k)-CA(k)
    # CA(k)-C(k)
    # Ca(k)-N(K+1) (except for C-term)
    # three angles to consider:
    # C(k-1)-N(k)-CA(k) (except for N-term)
    # N(k)-CA(k)-C(k)
    # CA(k)-C(k)-N(k+1) (except for C-term)
    # not sure whether to record C(k)-N(k+1)-CA(k+1) or C(k-1)-N(k)-CA(k)
    seqlen = len(mypose.sequence())
    result_dict = {}
    # gather xyz coords of all relevant atoms
    if k > 1:
        C_prev = mypose.residue(k-1).xyz("C")
    N_curr = mypose.residue(k).xyz("N")
    CA_curr = mypose.residue(k).xyz("CA")
    C_curr = mypose.residue(k).xyz("C")
    if k < seqlen:
        N_next = mypose.residue(k+1).xyz("N")
    # get relelvant atom-atom vectors
    if k > 1:
        CpNc = N_curr - C_prev
    NcCAc = CA_curr - N_curr
    CAcCc = C_curr - CA_curr
    if k < seqlen:
        CcNn = N_next - C_curr
    # get relevant bond lengths
    NcCAc_len = NcCAc.norm()
    result_dict["NcCAc_len"] = NcCAc_len
    CAcCc_len = CAcCc.norm()
    result_dict["CAcCc_len"] = CAcCc_len
    if k < seqlen:
        CcNn_len = CcNn.norm()
        result_dict["CcNn_len"] = CcNn_len
    # determine angles. There are three angles to consider:
    # C(k-1)-N(k)-CA(k) (except for N-term)
    if k > 1:
        CNCA = angle_between_vecs(CpNc.negated(), NcCAc)
        result_dict["CpNcCAc"] = CNCA
    # N(k)-CA(k)-C(k)
    NCAC = angle_between_vecs(NcCAc.negated(), CAcCc)
    result_dict["NcCAcCc"] = NCAC
    # CA(k)-C(k)-N(k+1) (except for C-term)
    if k < seqlen:
        CACN = angle_between_vecs(CAcCc.negated(), CcNn)
        result_dict["CAcCcNn"] = CACN
    return result_dict

def get_feature_matrix(mypose, padval=0):
    # GIVEN: a pose and a value for when lenghts and angles don't make
    # sense at the C and N terminus
    # RETURNS: a 2d numpy array
    # rows correspond to residues and columns correspond to features
    result = []
    column_names = ["NcCAc_len", "CAcCc_len", "CcNn_len", "CpNcCAc", "NcCAcCc", "CAcCcNn"]
    for res_pos in range(1,len(mypose.sequence())+1):
        feature_dict = get_bond_lengths_and_angles(mypose,res_pos)
        data_row = []
        # "zero padding"
        if res_pos == 1:
            feature_dict["CpNcCAc"] = padval
        if res_pos == len(mypose.sequence()):
            feature_dict["CcNn_len"] = padval
            feature_dict["CAcCcNn"] = padval
        for feature in column_names:
            data_row.append(feature_dict[feature])
        result.append(data_row)
    return np.array(result).T

def extractSS(pose):
    # Secondary structure term
    dssp = rosetta.core.scoring.dssp.Dssp(pose)
    dssp.insert_ss_into_pose(pose)
    _map = {"H":1, "L":2, "E":3}
    SS_mat = np.zeros((4, pose.size()))
    for ires in range(1, pose.size()+1):
        SS = pose.secstruct(ires)
        SS_mat[_map.get(SS, 0), ires-1] = 1
    return SS_mat

def extractOneBodyTerms(pose, padval=0):
    # All torsion angles in cosine/sine space
    # No transformation required
    
    # Get angles and and bond length
    bond_angles_lengths_mat = get_feature_matrix(pose, padval)
    features2 = ["NcCAc_len", "CAcCc_len", "CcNn_len", "CpNcCAc", "NcCAcCc", "CAcCcNn"]
    averages = [1.456790, 1.524227, 1.333378, 2.125835, 1.947459, 2.039060]
    bond_angles_lengths_mat = (bond_angles_lengths_mat.T-averages).T
    for i in range(len(features2)):
        bond_angles_lengths_mat[i] = np.tanh(bond_angles_lengths_mat[i])
        
    
    # 1 body energy terms
    score_terms = ["p_aa_pp", "rama_prepro", "omega", "fa_dun"]
    fa_scorefxn = get_fa_scorefxn()
    energy_term_mat = get_one_body_score_terms(pose, fa_scorefxn, score_terms)
    for i in range(len(score_terms)):
        if score_terms[i] != "fa_dun":
            energy_term_mat[i] = np.tanh(energy_term_mat[i])
        else:
            energy_term_mat[i] = np.arcsinh(energy_term_mat[i])-1
            
    # Secondary structure term
    SS_mat = extractSS(pose)
        
    return np.concatenate([bond_angles_lengths_mat, energy_term_mat, SS_mat]), features2+score_terms+["E", "L", "H"]

def process(args):
    filename, outfile, verbose = args
    if verbose: print("Processing", filename)
    try:
        pose = Pose()
        pose_from_file(pose, filename)
        fa_scorefxn = get_fa_scorefxn()
        score = fa_scorefxn(pose)

        # Features
        euler = pyp.getEulerOrientation(pose)
        maps = extract_multi_distance_map(pose)

        # Ivan Features
        pdict = init_pose(pose)
        idx = pdict['idx']
        val = pdict['val']
        phi = pdict['phi']
        psi = pdict['psi']
        omega6d = pdict['omega6d']
        theta6d = pdict['theta6d']
        phi6d = pdict['phi6d']

        # Nao Features 
        _2df, aas = extract_EnergyDistM(pose, energy_terms)
        _1df, _ = extractOneBodyTerms(pose)
        prop = extract_AAs_properties_ver1(aas)

        np.savez_compressed(outfile,
            idx = idx,
            val = val,
            phi = phi,
            psi = psi        ,
            omega6d = omega6d,
            theta6d = theta6d,
            phi6d = phi6d,
            tbt = _2df,
            obt = _1df,
            prop = prop,
            euler = euler,
            maps = maps)
    except:
        print(outfile, "Error: failed to process features.")

atypes = {}
types = {}
ntypes = 0
with open('/home/aivan/git/pgc3d/attempt04/data/groups20.txt', 'r') as f:
    data = csv.reader(f, delimiter=' ')
    for line in data:
        if line[1] in types:
            atypes[line[0]] = types[line[1]]
        else:
            types[line[1]] = ntypes
            atypes[line[0]] = ntypes
            ntypes += 1
            
def getData(tmp, mm, outfolder):
    data = np.load(tmp)
        
    # 3D information
    idx = data["idx"]
    val = data["val"]
    
    # 1D information
    angles = np.stack([np.sin(data["phi"]), np.cos(data["phi"]), np.sin(data["psi"]), np.cos(data["psi"])], axis=-1)
    obt = data["obt"].T
    prop = data["prop"].T
    
    # 2D information
    orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
    orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
    euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
    maps = data["maps"]
    tbt = data["tbt"].T
    sep = seqsep(tbt.shape[0])
    
    # Transformation
    tbt[:,:,0] = f(tbt[:,:,0])
    maps = f(maps)
    
    if mm:
        return (idx, val),\
                np.concatenate([angles, obt, prop], axis=-1),\
                np.concatenate([tbt, maps, euler, orientations, sep, np.load(join(outfolder,"dist.npy"))], axis=-1)
    else:
        return (idx, val),\
                np.concatenate([angles, obt, prop], axis=-1),\
                np.concatenate([tbt, maps, euler, orientations, sep], axis=-1)
    
# Sequence separtion features
def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def f(X, cutoff=6, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling  

def getDistribution(outfolder):
    path = outfolder
    tbts = [np.load(path+"/"+f)["tbt"][0,:,:] for f in os.listdir(path) if isfile(join(path,f)) and ".npz" in f]
    for i in range(len(tbts)-1):
        if not tbts[i].shape == tbts[i+1].shape:
            print("All pdbs in the input folder need to have the same size.")
    tbt = np.array(tbts)
    transformed = f(tbt, cutoff=6, scaling=1.0)
    digitization = np.arange(0.25,5.1,0.25)
    binned = np.eye(len(digitization)+1)[np.digitize(transformed, digitization)]
    normalized = np.sum(binned, axis=0)/tbt.shape[0]
    np.save(join(outfolder, "dist.npy"), normalized)

def main(args=None):
    # Parsing arguments
    if args == None:
        args = parse_opts(sys.argv[1:])

    # File existance checking
    if not isdir(args.infolder):
        print("Input folder does not exist.")
        return -1
    if not isdir(args.outfolder):
        print("Creating output folder.")
        os.mkdir(args.outfolder)
        
    base = "%s/models/"%SCRIPTPATH
    if args.multiDecoy:
        modelpath = base+"mmfull_adam00005_lddt10_aux033"
    else:
        modelpath = base+"smfull_adam00005_lddt10_aux033"
        
    if not args.noEnsemble:
        for i in range(1,5):
            if not isdir(modelpath+"_rep"+str(i)):
                print("Model checkpoint does not exist")
                return -1
    else:        
        if not isdir(modelpath+"_rep1"):
            print("Model checkpoint does not exist")
            return -1
        
    num_process = 1
    if args.process > 1:
        num_process = args.process
    
    # Loading pdb files
    samples = [i[:-4] for i in os.listdir(args.infolder) if i[-4:] == ".pdb"]
    if args.verbose: print("# samples:", len(samples))
    
    # Feature parsing
    inputs = [join(args.infolder, s)+".pdb" for s in samples]
    tmpoutputs = [join(args.outfolder, s)+".features.npz" for s in samples]
    arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
    
    # Parallelly process them
    if num_process == 1:
        for a in arguments:
            process(a)
    else:
        pool = multiprocessing.Pool(num_process)
        out = pool.map(process, arguments)
            
    # Get distribution features
    if args.multiDecoy:
        getDistribution(args.outfolder)

    # outputs
    lddts = {}
    
    # Making predictions
    lastmodel = 2 if args.noEnsemble else 5
    for i in range(1,lastmodel):
        modelname = modelpath+"_rep"+str(i)
        if args.verbose: print("Loading", modelname)
        if not args.multiDecoy:
            model = Model(obt_size=70,
                  tbt_size=33,
                  prot_size=None,
                  num_chunks=5,
                  optimizer="adam",
                  mask_weight=0.33,
                  lddt_weight=10.0,
                  name=modelname,
                  verbose=False)
        else:
            model = Model(obt_size=70,
                  tbt_size=54,
                  prot_size=None,
                  num_chunks=5,
                  optimizer="adam",
                  mask_weight=0.33,
                  lddt_weight=10.0,
                  name=modelname,
                  verbose=False)
        model.load()
        
        for j in range(len(tmpoutputs)):
            if args.verbose: print("Predicting for", samples[j], "(rep"+str(i)+")") 
            tmp = tmpoutputs[j] 
            batch = getData(tmp, args.multiDecoy, args.outfolder)
            lddt, estogram, mask = model.predict2(batch)
            if args.noEnsemble:
                lddts[samples[j]] = lddt
                np.savez_compressed(join(args.outfolder, samples[j]+".npz"),
                                    lddt = lddt,
                                    estogram = estogram,
                                    mask = mask)
            else:
                np.savez_compressed(join(args.outfolder, samples[j]+".rep"+str(i)+".npz"),
                                    lddt = lddt,
                                    estogram = estogram,
                                    mask = mask)
    if not args.noEnsemble:            
        # Merging predictions by taking average    
        for j in range(len(tmpoutputs)):
            
            # Loading predictions
            if args.verbose: print("Merging", samples[j])
            lddt = []
            estogram = []
            mask = []
            for i in range(1,5):
                temp = np.load(join(args.outfolder, samples[j]+".rep"+str(i)+".npz"))
                lddt.append(temp["lddt"])
                estogram.append(temp["estogram"])
                mask.append(temp["mask"])
                
            # Averaging
            lddt = np.mean(lddt, axis=0)
            estogram = np.mean(estogram, axis=0)
            mask = np.mean(mask, axis=0)
            lddts[samples[j]] = lddt
            
            # Saving
            np.savez_compressed(join(args.outfolder, samples[j]+".npz"),
                                lddt = lddt,
                                estogram = estogram,
                                mask = mask)
            
    # deleting...
    if not args.leavetemp:
        if args.multiDecoy:
            os.remove(join(args.outfolder, "dist.npy"))
        for i in range(len(tmpoutputs)):
            os.remove(tmpoutputs[i])
            if not args.noEnsemble:
                for j in range(1,5):
                    os.remove(join(args.outfolder, samples[i]+".rep"+str(j)+".npz"))
                    
    return lddts
            
if __name__== "__main__":
    main()
