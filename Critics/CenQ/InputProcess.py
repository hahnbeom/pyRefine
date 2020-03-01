#!/usr/bin/env python

import numpy as np
import scipy.spatial
import tensorflow as tf
from itertools import zip_longest
from .Sequence import *

BB_ATOMS = ['N', 'CA', 'C', 'O', 'CB']
MAX_TOT_RES_ACC   = 1000

def f_arcsinh(X, cutoff=6.0, scaling=3.0):
    X_prime = tf.maximum(X, tf.zeros_like(X) + cutoff) - cutoff
    return tf.asinh(X_prime)/scaling

def f_arcsinh_np(X, cutoff=6.0, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

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
    resNo_s = [[] for i in range(len(pdb_fn_s))]
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
                    resNo_s[i_pdb].append(resNo-1)
                    if resName == "GLY":
                        atom_read[-1, i_pdb, resNo-1, :] = xyz
    #
    atom_s = list()
    for i in range(len(BB_ATOMS)):
        atom_s.append(atom_read[i])
    atom_s.append(resNo_s)
    return atom_s 

class InputGenerator:
    def __init__(self, sess, **kwargs):
        self.sess = sess
        self.inputs = {'seq1hot':None,\
                       'blosum': None,\
                       'meiler': None,\
                       'msa': None,\
                       'relpos': None,\
                       'seqsep': None,\
                       'd_CB': None,\
                       'transf6d': None,\
                       'val': None,\
                       'idx': None,\
                       'BBtor': None}
        #
        with tf.variable_scope("InputGen"):
            self.build_graph()
    #
    def build_graph(self):
        with tf.name_scope("Input"):
            self.N  = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.CA = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.C  = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.O  = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.CB = tf.placeholder(tf.float32, shape=[None, None, 3])
            #
            # for BBtor calc
            self.N_next = tf.placeholder(tf.float32, shape=[None, None, 3])
            self.C_prev = tf.placeholder(tf.float32, shape=[None, None, 3])
            #
            self.mask = tf.placeholder(tf.float32, shape=[None, None, None])
            self.atypes = tf.placeholder(tf.int8, shape=[None,None])
        #
        self.d_CB = self.calc_distance(self.CB, self.CB, self.mask, transform=True) # common input feature for all ML models
        #
        self.BBtor = self.calc_BBtors(self.C_prev, self.N, self.CA, self.C, self.N_next)
        self.transf6d = self.calc_6d_transforms(self.N, self.CA, self.C, self.mask)
        self.idx, self.val = self.set_neighbors3D_coarse(self.N, self.CA, self.C, self.O, self.CB, self.mask, self.atypes)

    def calc_distance(self, A, B, mask, transform=False, mask_to_negative=False):
        # A & B: (n_conf, n_res, 3)
        # D = row_norms_A - 2*A*B + row_norms_B
        n_str = tf.shape(A)[0]
        n_res_A = tf.shape(A)[1]
        n_res_B = tf.shape(B)[1]
        #
        row_norms_A = tf.reduce_sum(tf.square(A), axis=-1)
        row_norms_A = tf.reshape(row_norms_A, [n_str, n_res_A, 1])
        #
        row_norms_B = tf.reduce_sum(tf.square(B), axis=-1)
        row_norms_B = tf.reshape(row_norms_B, [n_str, 1, n_res_B])
        #
        D = row_norms_A - 2 * tf.matmul(A, tf.transpose(B, perm=(0,2,1))) + row_norms_B # squared distance
        D = tf.maximum(D, 0.0)
        D = tf.sqrt(D) # (n_str, n_res_A, n_res_B)
        #
        if transform:
            return f_arcsinh(D)*mask
        else:
            if mask_to_negative:
                negatives = tf.fill(tf.shape(mask), -1.0) 
                return D*tf.where(mask < 1.0, negatives, mask)
            return D*mask

    def get_dihedrals(self, A, B, C, D):
        if len(A.shape) > 2:
            A = tf.reshape(A, [-1, 3])
            B = tf.reshape(B, [-1, 3])
            C = tf.reshape(C, [-1, 3])
            D = tf.reshape(D, [-1, 3])
        #
        B0 = -1.0*(B - A) # n_conf*n_res, 3
        B1 = C - B
        B2 = D - C
        #
        B1 /= tf.linalg.norm(B1, axis=-1)[:,None]
        #
        V = B0 - tf.reduce_sum(B0*B1, axis=-1)[:,None]*B1
        W = B2 - tf.reduce_sum(B2*B1, axis=-1)[:,None]*B1
        #
        X = tf.reduce_sum(V*W, axis=-1)
        Y = tf.reduce_sum(tf.cross(B1, V)*W, axis=-1)
        #
        ang_raw = tf.atan2(Y, X)
        ang = tf.where(tf.is_nan(ang_raw), tf.zeros_like(ang_raw), ang_raw)
        return ang 

    def get_angles(self, A, B, C):
        if len(A.shape) > 2:
            A = tf.reshape(A, [-1, 3])
            B = tf.reshape(B, [-1, 3])
            C = tf.reshape(C, [-1, 3])
        V = A - B
        V /= tf.linalg.norm(V, axis=-1)[:,None] # normalized vector

        W = C - B
        W /= tf.linalg.norm(W, axis=-1)[:,None]
        
        X = tf.reduce_sum(V*W, axis=-1) # dot product v dot w = |v||w|cos(theta)

        ang_raw = tf.acos(X)
        ang = tf.where(tf.is_nan(ang_raw), tf.zeros_like(ang_raw), ang_raw)
        return ang

    def get_virtual_CB(self, N, CA, C):
        b = CA - N
        c = C - CA
        a = tf.cross(b, c)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
        return Cb

    def set_local_frame(self, N, CA, CB):
        # center at Ca, and put Cb on to the z-axis
        z = CB - CA # (n_str, n_res, 3)
        z /= tf.linalg.norm(z, axis=-1)[:,:,None] # (n_str, n_res, 3)
        #
        x = tf.linalg.cross(CA-N, z)
        x /= tf.linalg.norm(x, axis=-1)[:,:,None]
        #
        y = tf.linalg.cross(z, x)
        y /= tf.linalg.norm(y, axis=-1)[:,:,None]
        #
        xyz = tf.stack([x,y,z]) # (3, n_str, n_res, 3)
        return tf.transpose(xyz, [1,2,0,3])

    def set_neighbors3D_coarse(self, N, CA, C, O, CB, mask_BB, atypes, d_max=14.0):
        # parameters
        nbins = 24
        #nbins = 24.0
        width = 19.2
        h = width / (nbins-1) # bin size
        #
        n_res = tf.shape(CA)[1]
        virt_Cb = self.get_virtual_CB(N, CA, C)
        lfr = self.set_local_frame(N, CA, virt_Cb)
        #
        xyz = tf.stack((N, CA, C, O, CB), axis=-2) #(n_batch, n_res, 5, 3)
        xyz = tf.reshape(xyz, (-1, n_res*5, 3))
        mask_xyz = tf.concat((mask_BB, mask_BB, mask_BB, mask_BB, mask_BB), axis=-1) #(n_batch, n_res, n_res*5)
        dist = self.calc_distance(CA, xyz, mask_xyz, mask_to_negative=True) # (n_batch, n_res, n_res*5)
        #
        indices = tf.where(tf.logical_and(dist <= d_max, dist >= 0.0)) # [# of true, 3], 0 for batch, 1 for center CA, 2 for others
        #
        idx_CA = indices[:,:2] # (N, 2)
        idx_all = tf.stack((indices[:,0], indices[:,2]), axis=-1) # (N, 2)
        N = indices.shape[0] # total number of neighbors
        #
        types = tf.cast(tf.gather_nd(atypes, idx_all), dtype=tf.int32) # atypes: (n_str, n_res*5) => types: (N,)
        self.neigh = tf.stack((indices[:,1], indices[:,2], tf.cast(types, dtype=tf.int64)), axis=-1)
        xyz_env = tf.gather_nd(xyz, idx_all) # (N, 3)
        #
        xyz_ca = tf.gather_nd(CA, idx_CA) # (N, 3)
        lfr_ca = tf.gather_nd(lfr, idx_CA) # (N, 3, 3)
        #
        xyz_shift = xyz_env - xyz_ca # (N, 3)
        xyz_new = tf.reduce_sum(lfr_ca * xyz_shift[:,None,:], axis=-1) # (N, 3)
        xyz_new = tf.transpose(xyz_new) # (3, N)
        #
        # shift all contacts to the center of the box and scale the coordinates by h
        xyz = (xyz_new + 0.5*width)/h # (3, N)
        #
        # discretized xyz coordinates
        klm = tf.floor(xyz) # (3, N)
        d_0 = xyz - klm
        d_1 = 1.0-d_0
        klm = tf.cast(klm, dtype=tf.int32)
        #
        # trilinear interpolation
        klm0 = klm[0,:]
        klm1 = klm[1,:]
        klm2 = klm[2,:]
        #
        V000 = d_0[0,:]*d_0[1,:]*d_0[2,:]
        V100 = d_1[0,:]*d_0[1,:]*d_0[2,:]
        V010 = d_0[0,:]*d_1[1,:]*d_0[2,:]
        V110 = d_1[0,:]*d_1[1,:]*d_0[2,:]
        #
        V001 = d_0[0,:]*d_0[1,:]*d_1[2,:]
        V101 = d_1[0,:]*d_0[1,:]*d_1[2,:]
        V011 = d_0[0,:]*d_1[1,:]*d_1[2,:]
        V111 = d_1[0,:]*d_1[1,:]*d_1[2,:]
        #
        idx_CA = tf.cast(tf.transpose(idx_CA), dtype=tf.int32)
        #
        a000 = tf.stack([idx_CA[0], idx_CA[1], klm0,   klm1,   klm2, types], axis=-1) # (batch_idx, res_idx, ....)
        a100 = tf.stack([idx_CA[0], idx_CA[1], klm0+1, klm1,   klm2, types], axis=-1)
        a010 = tf.stack([idx_CA[0], idx_CA[1], klm0,   klm1+1, klm2, types], axis=-1)
        a110 = tf.stack([idx_CA[0], idx_CA[1], klm0+1, klm1+1, klm2, types], axis=-1)
        #
        a001 = tf.stack([idx_CA[0], idx_CA[1],  klm0,   klm1,   klm2+1, types], axis=-1)
        a101 = tf.stack([idx_CA[0], idx_CA[1],  klm0+1, klm1,   klm2+1, types], axis=-1)
        a011 = tf.stack([idx_CA[0], idx_CA[1],  klm0,   klm1+1, klm2+1, types], axis=-1)
        a111 = tf.stack([idx_CA[0], idx_CA[1],  klm0+1, klm1+1, klm2+1, types], axis=-1)
        #
        a = tf.concat([a000, a100, a010, a110, a001, a101, a011, a111], axis=0) # (8*N, 6)
        V = tf.concat([V111, V011, V101, V001, V110, V010, V100, V000], axis=0) # (8*N, 1)
        #
        condition = tf.logical_and(tf.reduce_min(a[:,2:5], axis=-1) >= 0, tf.reduce_max(a[:,2:5], axis=-1) < nbins) # fit into box
        condition = tf.logical_and(condition, a[:,5] >= 0) # types are valid
        condition = tf.logical_and(condition, V > 1e-5) # values are large enough
        fit_into_box = tf.where(condition)
        #
        a = tf.gather_nd(a, fit_into_box)
        b = tf.gather_nd(V, fit_into_box)
        #
        return tf.cast(a, dtype=tf.uint16), tf.cast(b, dtype=tf.float16)

    def calc_6d_transforms(self, N, CA, C, mask_in, d_max=20.0):
        mask = mask_in * (tf.ones_like(mask_in) - tf.matrix_band_part(mask_in, 0, 0))
        Cb = self.get_virtual_CB(N, CA, C)
        # calc 6D transforms for all pairs within d_max
        dist = self.calc_distance(Cb, Cb, mask)
        idx = tf.where(tf.logical_and(dist < d_max, dist > 0.0)) # [# of true, 3], 0 for batch, 1 for res_1, 2 for res_2
        #
        idx_1 = idx[:,:2] # (N, 2)
        idx_2 = tf.stack((idx[:,0], idx[:,2]), axis=-1)
        #
        CA_1 = tf.gather_nd(CA, idx_1)
        CA_2 = tf.gather_nd(CA, idx_2)
        #
        Cb_1 = tf.gather_nd(Cb, idx_1)
        Cb_2 = tf.gather_nd(Cb, idx_2)
        #
        N_1 = tf.gather_nd(N, idx_1)
        N_2 = tf.gather_nd(N, idx_2)
        # 
        #omega6d = tf.zeros_like(dist) # (n_str, n_res, n_res)
        #theta6d = tf.zeros_like(dist)
        #phi6d = tf.zeros_like(dist)
        #
        ang = self.get_dihedrals(CA_1, Cb_1, Cb_2, CA_2)
        omega6d = tf.scatter_nd(idx, ang, tf.shape(dist, out_type=tf.int64))
        #
        ang = self.get_dihedrals(N_1, CA_1, Cb_1, Cb_2)
        theta6d = tf.scatter_nd(idx, ang, tf.shape(dist, out_type=tf.int64))
        #
        ang = self.get_angles(CA_1, Cb_1, Cb_2)
        phi6d = tf.scatter_nd(idx, ang, tf.shape(dist, out_type=tf.int64))
        #
        orientations = tf.stack((omega6d, theta6d, phi6d), axis=-1)
        orientations = orientations * mask[:,:,:,None]
        #
        dist = dist[:,:,:,np.newaxis]*mask[:,:,:,None]
        orien = tf.concat((tf.sin(orientations), tf.cos(orientations)), axis=-1)
        return tf.concat((dist, orien), axis=-1)

    def calc_BBtors(self, C_prev, N, CA, C, N_next):
        # get 1d mask (masking chain break)
        n_batch = tf.shape(N)[0]
        n_res = tf.shape(N)[1]
        #
        breaks = tf.linalg.norm(N[:,:,:] - C_prev[:,:-1,:], axis=-1)
        breaks = tf.where(tf.is_nan(breaks), tf.zeros_like(breaks), breaks)
        mask_phi = tf.ones_like(breaks)
        mask_phi = tf.where(breaks > 2.5, tf.zeros_like(mask_phi), mask_phi)
        #
        breaks = tf.linalg.norm(N_next[:,1:,:] - C, axis=-1)
        breaks = tf.where(tf.is_nan(breaks), tf.zeros_like(breaks), breaks)
        mask_psi = tf.ones_like(breaks)
        mask_psi = tf.where(breaks > 2.5, tf.zeros_like(mask_psi), mask_psi)
        #
        # convert to tensor
        phi = self.get_dihedrals(C_prev[:,:-1,:], N, CA, C)
        psi = self.get_dihedrals(N, CA, C, N_next[:,1:,:])
        #
        phi = phi*tf.reshape(mask_phi, [-1])
        psi = psi*tf.reshape(mask_psi, [-1])
        #
        phi = tf.reshape(phi, (n_batch, n_res))
        psi = tf.reshape(psi, (n_batch, n_res))
        #
        return tf.stack((tf.sin(phi), tf.cos(phi), tf.sin(psi), tf.cos(psi)), axis=-1)

    def relpos_and_seqsep(self, nres_s): # faster on CPU (400us for a 150aa protein, 100 ms for 50 proteins with 200aa)
        min_res = np.min(nres_s)
        max_res = np.max(nres_s)
        #
        if max_res == min_res:
            pos = np.arange(max_res)
            relpos = np.minimum(pos, max_res-pos)*1.0/max_res * 2.0
            tiled_pos = np.tile(pos, (max_res,1))
            seqsep = np.abs(tiled_pos - tiled_pos.T) / 100.0 - 1.0
            #
            relpos = np.broadcast_to(relpos, (len(nres_s), max_res))
            seqsep = np.broadcast_to(seqsep, (len(nres_s), max_res, max_res))
        else:
            pos = [np.arange(nres) for nres in nres_s]
            zipped = np.array(list(itertools.zip_longest(*pos, fillvalue=np.nan))).T

            relpos = np.minimum(zipped, nres_s[:,None]-zipped)*1.0/nres_s[:,None] * 2.0
            relpos[np.isnan(relpos)] = 0.0

            tiled_pos = np.tile(zipped[:,None,:], (1, max_res, 1))
            seqsep = np.abs(tiled_pos - np.transpose(tiled_pos, (0,2,1))) / 100.0 - 1.0
            seqsep[np.isnan(seqsep)] = 0.0
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

    def process(self, pose_s, seq_fn=None, seq=None, update_seq=False):
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
        if (update_seq) or (self.inputs['seq1hot'] == None):
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
            self.inputs['relpos'], self.inputs['seqsep'] = self.relpos_and_seqsep(np.array([n_res]))
            #
            self.inputs['aa_prop'] = np.concatenate((self.inputs['seq1hot'][None,:,:], self.inputs['blosum'][None,:,:],\
                                                     self.inputs['relpos'][:,:,None], self.inputs['meiler'][None,:,:]), axis=-1)
        n_res = len(self.inputs['seq1hot'])
        #
        n_batch = len(pose_s)
        if is_file:
            # Read input pdbs & get backbone + CB coords
            N, CA, C, O, CB, resNo_all = get_coords_pdb(pdb_fn_s, n_res) # read N, CA, C, CB coordinates
        else:
            N = list()
            CA = list()
            C = list()
            O = list()
            CB = list()
            resNo_all = list()
            #
            for pose in pose_s:
                N.append(np.array([pose.residue(i).xyz('N') for i in range(1, n_res+1)]))
                CA.append(np.array([pose.residue(i).xyz('CA') for i in range(1, n_res+1)]))
                C.append(np.array([pose.residue(i).xyz('C') for i in range(1, n_res+1)]))
                O.append(np.array([pose.residue(i).xyz('O') for i in range(1, n_res+1)]))
                CB.append(np.array([pose.residue(i).xyz('CB') if pose.residue(i).has('CB') else pose.residue(i).xyz('CA') for i in range(1, n_res+1)]))
                #
                resNo_all.append(np.array([pose.pdb_info().number(i) for i in range(1, n_res+1)])-1)
            N = np.stack(N, axis=0)
            CA = np.stack(CA, axis=0)
            C = np.stack(C, axis=0)
            O = np.stack(O, axis=0)
            CB = np.stack(CB, axis=0)
        #
        C_prev = np.insert(C, 0, np.nan, axis=1)
        N_next = np.insert(N, n_res, np.nan, axis=1)
        # mask for missing residues in input pdb
        mask = np.zeros((n_batch, n_res, n_res), dtype=np.float32) # mask all missing
        for i_pdb in range(n_batch):
            mask[np.ix_([i_pdb], resNo_all[i_pdb], resNo_all[i_pdb])] = 1.0
        #
        output_keys = ['d_CB', 'BBtor', 'transf6d', 'idx', 'val']
        ops_to_run = [self.d_CB, self.BBtor, self.transf6d, self.idx, self.val]
        #
        inputs = {self.CB: CB, self.mask: mask}
        inputs[self.N] = N
        inputs[self.C] = C
        inputs[self.CA] = CA
        inputs[self.O] = O
        inputs[self.C_prev] = C_prev
        inputs[self.N_next] = N_next
        #
        atypes = np.full((n_batch, n_res, 5), -1, dtype=np.int8)
        seq_in_number = self.inputs['msa'][0]
        for i_pdb in range(n_batch):
            atypes[i_pdb, :, :] = rosetta_atom_types[np.ix_(seq_in_number, np.arange(5))]
        inputs[self.atypes] = np.reshape(atypes, (-1, n_res*5))
        #
        outputs = self.sess.run(ops_to_run, feed_dict=inputs)
        for i_key, key in enumerate(output_keys):
            self.inputs[key] = outputs[i_key]

        # Input for CenQ estimation
        # Input:
        #  - f1d: BBtor, aa properties (seq, blosum, relpos, meiler)
        #  - f2d: dCb, transf6d, seqsep

        # make obt, tbt
        nbatch, nres, _ = self.inputs['BBtor'].shape
        aa_prop = np.broadcast_to(self.inputs['aa_prop'], (nbatch, nres, self.inputs['aa_prop'].shape[-1]))
        obt = np.concatenate((self.inputs['BBtor'], aa_prop), axis=-1)
        #
        seqsep = np.broadcast_to(self.inputs['seqsep'], (nbatch, nres, nres))
        #
        tbt = np.zeros((8, nbatch, nres, nres), dtype=np.float32) # less computation time
        transf6d = np.transpose(self.inputs['transf6d'], (3,0,1,2))
        tbt[0,:,:,:] = self.inputs['d_CB']
        tbt[-7:-1,:,:,:] = transf6d[1:]
        tbt[-1,:,:,:] = seqsep
        tbt = np.transpose(tbt, (1,2,3,0))
        
        batch_s = list()
        n_prot, n_res = self.inputs['d_CB'].shape[:2]
        #
        prot_in_batch = MAX_TOT_RES_ACC // n_res
        if n_prot % prot_in_batch == 0:
            n_batch = n_prot // prot_in_batch
        else:
            n_batch = n_prot // prot_in_batch + 1
       
        for i_b in range(n_batch):
            start = i_b*prot_in_batch
            end   = (i_b+1)*prot_in_batch
            #
            indices = np.where((self.inputs['idx'][:,0] >= start) & (self.inputs['idx'][:,0] < end))
            idx = self.inputs['idx'][indices]
            idx[:,0] -= start
            val = self.inputs['val'][indices]
            for_3d = (idx, val)

            batch = [for_3d, obt[start:end], tbt[start:end]]
            batch_s.append(batch)
        
        return batch_s
