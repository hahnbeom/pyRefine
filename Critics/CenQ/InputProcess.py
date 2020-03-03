#!/usr/bin/env python

import os
gpu_name = os.popen("nvidia-smi --list-gpus").readlines()[0]
import time
import numpy as np
import scipy.spatial
import tensorflow as tf
#print (tf.test.gpu_device_name())
from itertools import zip_longest
from .Sequence import *

BB_ATOMS = ['N', 'CA', 'C', 'O', 'CB']
MAX_TOT_RES_ACC   = 180000 # (60^2) * 50: rtx2080 can process 50 decoys having 60aa at same time.
if "TITAN" in gpu_name:
    MAX_TOT_RES_ACC = MAX_TOT_RES_ACC * 3
elif "QUADRO" in gpu_name:
    MAX_TOT_RES_ACC = MAX_TOT_RES_ACC * 6

def get_seq_from_pdb(pdb_fn):
    seq = ''
    prev_resNo = None
    with open(pdb_fn) as fp:
        for line in fp:
            if not line.startswith("ATOM"): continue
            resNo = int(line[22:26])
            atmName = line[12:16].strip()
            #
            if atmName != "CA": continue
            if resNo != prev_resNo:
                resName = line[17:20]
                seq += three_to_one[resName]
            prev_resNo = resNo
    return seq

def get_coords_pdb(pdb_fn_s, n_res):
    atom_read = np.zeros((4, len(pdb_fn_s), n_res, 3), dtype=np.float32)
    #
    for i_pdb, pdb_fn in enumerate(pdb_fn_s):
        with open(pdb_fn) as fp:
            for line in fp:
                if not line.startswith("ATOM"): continue
                resNo = int(line[22:26])
                resName = line[17:20]
                atmName = line[12:16].strip()
                #
                if atmName in BB_ATOMS:
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    atom_read[BB_ATOMS.index(atmName), i_pdb, resNo-1, :] = xyz
                #
                if atmName == "CA":
                    if resName == "GLY":
                        atom_read[-1, i_pdb, resNo-1, :] = xyz
    #
    atom_s = list()
    for i in range(len(BB_ATOMS)):
        atom_s.append(atom_read[i])
    return atom_s 

class InputGenerator:
    def __init__(self, **kwargs):
        self.inputs = {'seq1hot':None,\
                       'blosum': None,\
                       'meiler': None,\
                       'msa': None,\
                       'relpos': None,\
                       'seqsep': None}
    
    def relpos_and_seqsep(self, n_res): # faster on CPU (400us for a 150aa protein, 100 ms for 50 proteins with 200aa)
        pos = np.arange(n_res)
        relpos = np.minimum(pos, n_res-pos)*1.0/n_res * 2.0
        tiled_pos = np.tile(pos, (n_res,1))
        seqsep = np.abs(tiled_pos - tiled_pos.T) / 100.0 - 1.0
        #
        return relpos, seqsep

    def get_1d_seq_features(self, seq_fn=None, seq=None):
        import string
        seqs = []
        #
        if seq != None:
            seqs.append(seq)
        elif seq_fn != None:
            table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

            # read file
            with open(seq_fn) as fp:
                for line in fp:
                    if line[0] == '>': continue
                    seqs.append(line.rstrip().translate(table)) # remove lowercase letters
        #
        # convert letters into numbers
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
        msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            msa[msa == alphabet[i]] = i

        msa[msa > 20] = 0 #VIRTUAL RES: ALANINE 
        #msa[msa > 20] = 20
        #
        seq = msa[0]
        return np.eye(20)[seq], blosum(seq), meiler(seq), msa # Note: it returns [n_seq, n_res] MSA results

    def process(self, pose_s, seq_fn=None, seq=None, update_seq=False, distogram=None):
        start_time = time.time()
        # Get sequence-based features
        # 1. 1-hot encoded AA sequence
        # 2. Blosum6d matrix
        # 3. meiler features
        # 4. MSA (for SStor)
        # 5. relpos (for Accuracy)
        # 6. seqsep (for Accuracy)
        #
        # check input pose_s is pdb files or rosetta poses
        is_file = False
        if isinstance(pose_s[0], str):
            is_file = True


        n_res = None
        if (update_seq) or (np.any(self.inputs['seq1hot'] == None)):
            if seq_fn != None: # get sequence info from file (input sequence fasta file or a3m file)
                self.inputs['seq1hot'], self.inputs['blosum'], self.inputs['meiler'], self.inputs['msa'] = self.get_1d_seq_features(seq_fn=seq_fn)
            elif seq != None:
                self.inputs['seq1hot'], self.inputs['blosum'], self.inputs['meiler'], self.inputs['msa'] = self.get_1d_seq_features(seq=seq)
            else:
                if is_file:
                    seq = get_seq_from_pdb(pose_s[0])
                else:
                    seq = pose_s[0].sequence()
                    while seq.endswith("X"): 
                        seq = seq[:-1] #TAKE CARE OF VROOT AT C-TERM -- may not work if vres at middle 
                self.inputs['seq1hot'], self.inputs['blosum'], self.inputs['meiler'], self.inputs['msa'] = self.get_1d_seq_features(seq=seq)
            #
            n_res = len(self.inputs['seq1hot'])
            self.inputs['relpos'], self.inputs['seqsep'] = self.relpos_and_seqsep(n_res)
            #
            self.inputs['aa_prop'] = np.concatenate((self.inputs['seq1hot'], self.inputs['blosum'],\
                                                     self.inputs['relpos'][:,None], self.inputs['meiler']), axis=-1)
        n_res = len(self.inputs['seq1hot'])
        #
        atypes = np.full((n_res, 5), -1, dtype=np.int8)
        seq_in_number = self.inputs['msa'][0]
        atypes[:, :] = rosetta_atom_types[np.ix_(seq_in_number, np.arange(5))]
        atypes = atypes.flatten()
        #
        n_batch = len(pose_s)
        if is_file:
            # Read input pdbs & get backbone + CB coords
            N, CA, C, O, CB = get_coords_pdb(pdb_fn_s, n_res) # read N, CA, C, CB coordinates
        else:
            N = np.zeros((len(pose_s), n_res, 3), dtype=np.float64)
            CA = np.zeros((len(pose_s), n_res, 3), dtype=np.float64)
            C = np.zeros((len(pose_s), n_res, 3), dtype=np.float64)
            O = np.zeros((len(pose_s), n_res, 3), dtype=np.float64)
            CB = np.zeros((len(pose_s), n_res, 3), dtype=np.float64)
            #
            for i_p, pose in enumerate(pose_s):
                for i in range(1, n_res+1):
                    N_xyz = pose.residue(i).xyz('N')
                    CA_xyz = pose.residue(i).xyz('CA')
                    C_xyz = pose.residue(i).xyz('C')
                    O_xyz = pose.residue(i).xyz('O')
                    CB_xyz = pose.residue(i).xyz('CB') if pose.residue(i).has('CB') else pose.residue(i).xyz('CA')
                    for k in range(3):
                        N[i_p, i-1, k] = N_xyz[k]
                        CA[i_p, i-1, k] = CA_xyz[k]
                        C[i_p, i-1, k] = C_xyz[k]
                        O[i_p, i-1, k] = O_xyz[k]
                        CB[i_p, i-1, k] = CB_xyz[k]
        #
        # Input for CenQ estimation
        # Input:
        #  - f1d: BBtor, aa properties (seq, blosum, relpos, meiler)
        #  - f2d: dCb, transf6d, seqsep

        # make obt, tbt
        n_prot, n_res, _ = N.shape
        
        batch_s = list()
        #
        prot_in_batch = MAX_TOT_RES_ACC // (n_res**2)
        if n_prot % prot_in_batch == 0:
            n_batch = n_prot // prot_in_batch
        else:
            n_batch = n_prot // prot_in_batch + 1
       
        for i_b in range(n_batch):
            start = i_b*prot_in_batch
            end   = (i_b+1)*prot_in_batch
            #
            batch = [N[start:end], CA[start:end], C[start:end], O[start:end], CB[start:end]]
            batch_s.append(batch)
        
        return self.inputs['aa_prop'], self.inputs['seqsep'], atypes, batch_s
