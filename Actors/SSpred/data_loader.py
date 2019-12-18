#!/usr/bin/env python3

import os
import glob
import numpy as np
import scipy
from scipy.special import i0
#import tensorflow as tf
from pyrosetta import *

init("-mute all")

#============================================
# Global variables such as data directory path
#============================================
MSA_DIR = "/home/minkbaek/DeepLearn/torsion/msa"
LABEL_dir = "/home/minkbaek/DeepLearn/torsion/labels"
DECOY_DIR = "/home/minkbaek/DeepLearn/torsion/w_6Dstr/1D_after_2D/data"

def get_labels(pdb):
    label_fn = os.path.join(LABEL_dir, pdb + ".npz")
    data = np.load(label_fn)
    #
    n_res = data['dssp'].shape[0]
    dist_cb = np.reshape(np.eye(32)[np.reshape(data['dist_cb'], [-1])], [n_res, n_res, -1])
    dist_ct = np.reshape(np.eye(32)[np.reshape(data['dist_ct'], [-1])], [n_res, n_res, -1])
    dist_cat = np.reshape(np.eye(32)[np.reshape(data['dist_cat'], [-1])], [n_res, n_res, -1])
    dist_tip = np.reshape(np.eye(32)[np.reshape(data['dist_tip'], [-1])], [n_res, n_res, -1])
    return data['dssp'], data['phi'], data['psi'], data['omg'], dist_cb, dist_ct, dist_cat, dist_tip 

def read_preprocessed_decoy(filename, random_pick=False, tag=None):
    if not isinstance(filename, str):
        pre_dir = os.path.join(DECOY_DIR, filename.decode('utf-8'))
    else:
        pre_dir = os.path.join(DECOY_DIR, filename)
    #
    fn_s = sorted(glob.glob("%s/*.npz"%pre_dir))
    if random_pick:
        fn = np.random.choice(fn_s, 1)[0]
    else:
        fn = "%s/%s.al_0001.npz"%(pre_dir, tag)
    #
    decoy_data = np.load(fn)
    return decoy_data['bb_angs'], decoy_data['str_feat']

def subsample_msa(msa):
    n_sub = int(msa.shape[0]/2)
    if n_sub < 5:
        return msa
    seq = msa[0]
    tmp = msa[1:]
    np.random.shuffle(tmp)
    return np.concatenate([seq[np.newaxis,:], tmp[:n_sub-1,:]], axis=0)

def prepare_seq_feat_target(pdb):
    msa_fn = os.path.join(MSA_DIR, pdb+".npy")
    msa = np.load(msa_fn)
    #
    seq = msa[0]
    sub_msa = subsample_msa(msa)
    #
    return seq, sub_msa

def load_train_data(pdb, decoy_tag, random_pick=False):
    #
    seq, msa = prepare_seq_feat_target(pdb)
    #
    SS_label, phi_label, psi_label, omg_label, dist_cb, dist_cbt, dist_cat, dist_tip = get_labels(pdb)
    #
    bb_angs, str_feat = read_preprocessed_decoy(pdb, random_pick=random_pick, tag=decoy_tag)
    #
    # normalize distance map (0~1)
    str_feat[0] = str_feat[0]/20.0
    #
    n_res = len(seq)
    mask = np.ones(int(n_res), dtype=np.float32)
    mask[0] = 0.0
    mask[-1] = 0.0
    return seq, msa, bb_angs, str_feat, SS_label, phi_label, psi_label, omg_label, dist_cb, dist_cbt, dist_cat, dist_tip, mask

def get_backbone_angles(pdb_fn):
    model = pose_from_file(pdb_fn)
    n_res = rosetta.core.pose.nres_protein(model)
    #
    phi = np.array(np.deg2rad([model.phi(i) for i in range(1, n_res+1)])).astype(np.float32)    
    psi = np.array(np.deg2rad([model.psi(i) for i in range(1, n_res+1)])).astype(np.float32)    
    omg = np.array(np.deg2rad([model.omega(i) for i in range(1, n_res+1)])).astype(np.float32)
    #
    return np.array([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi), np.cos(omg), np.sin(omg)]).T, model

def get_dihedrals(a, b, c, d):
    b0 = -1.0*(b-a)
    b1 = c-b
    b2 = d-c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y,x)

def get_angles(a, b, c):
    v = a-b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c-b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v*w, axis=1)
    
    return np.arccos(x)

def get_neighbors(pose, dmax):
    nres = rosetta.core.pose.nres_protein(pose)
    #
    N = np.stack([np.array(pose.residue(i).atom('N').xyz()) for i in range(1,nres+1)])
    CA = np.stack([np.array(pose.residue(i).atom('CA').xyz()) for i in range(1,nres+1)])
    C = np.stack([np.array(pose.residue(i).atom('C').xyz()) for i in range(1,nres+1)])
    #
    # Virtual CB given N, CA, C
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
    #
    # neighbors search
    kdCB = scipy.spatial.cKDTree(CB)
    indices = kdCB.query_ball_tree(kdCB, dmax)
    #
    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # CB-CB dist matrix
    dist = np.zeros((nres, nres))
    dist[idx0, idx1] = np.linalg.norm(CB[idx1]-CB[idx0], axis=-1)
    #
    # omega angle = CA-CB-CB-CA dihedrals
    omega = np.zeros((nres, nres))
    omega[idx0, idx1] = get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1])

    # Theta
    theta = np.zeros((nres, nres))
    theta[idx0, idx1] = get_dihedrals(N[idx0], CA[idx0], CB[idx0], CB[idx1])

    # phi
    phi = np.zeros((nres, nres))
    phi[idx0, idx1] = get_angles(CA[idx0], CB[idx0], CB[idx1])
    
    return dist, omega, theta, phi

def get_2D_str_feat(model):
    dist, omega_raw, theta_asym_raw, phi_asym_raw = get_neighbors(model, 20.0)
    dist = np.reshape(dist, (1, dist.shape[0], dist.shape[1]))
    phi_asym = np.array([np.cos(phi_asym_raw), np.sin(phi_asym_raw)])
    theta_asym = np.array([np.cos(theta_asym_raw), np.sin(theta_asym_raw)])
    omega = np.array([np.cos(omega_raw), np.sin(omega_raw)])
   
    print (dist.shape)
    print (phi_asym.shape)
    print (theta_asym.shape)
    print (omega.shape)
    str_feat = np.concatenate((dist, phi_asym, theta_asym, omega), axis=0)
    str_feat = str_feat.astype(np.float32)
    str_feat = np.moveaxis(str_feat, 0, -1)
    
    return str_feat

def read_msa(a3m_fn):
    import string

    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file
    with open(a3m_fn) as fp:
        for line in fp:
            if line[0] == '>': continue
            seqs.append(line.rstrip().translate(table)) # remove lowercase letters
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    msa[msa > 20] = 20
    return msa


def make_input_features(a3m_fn, pdb_fn):
    msa = read_msa(a3m_fn)
    seq = msa[0]
    #
    bb_angs, model = get_backbone_angles(pdb_fn)
    #
    feat_2D = get_2D_str_feat(model)
    feat_2D[0] = feat_2D[0]/20.0

    return seq, msa, bb_angs, feat_2D


## test
##for fn in [line.strip() for line in open('train.list')]:
##    feat = prepare_seq_feats(fn)
##    print (feat)
#    
#train_data = train_dataset('train.list')
#iterator = train_data.make_initializable_iterator()
#next_element = iterator.get_next()
#
#with tf.Session() as sess:
#    sess.run(iterator.initializer)
#    print (sess.run(next_element))
