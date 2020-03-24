#!/usr/bin/env python

import os
import json
import numpy as np
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from .CenQ_resnet import build_resnet
from .InputProcess import InputGenerator
SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))

def f_arcsinh(X, cutoff=6.0, scaling=3.0):
    X_prime = tf.maximum(X, tf.zeros_like(X) + cutoff) - cutoff
    return tf.asinh(X_prime)/scaling

class CenQPredictor:
    def __init__(self, sess, ver=1):
        self.sess = sess
        self.ver = ver
        #
        self.prot_size = None
        self.batch_size = None
        self.nretype = 20
        self.num_chunks = 5
        #
        if ver == 1:
            self.obt_size = 56 
            self.tbt_size = 45
            self.ignore3dconv = False
        else:
            self.obt_size = 56 
            self.tbt_size = 8
            self.ignore3dconv = False
        #
        self.ops = self.build_graph()
        self.restore_model("%s/models_cenQ.v%d/model.ckpt"%(SCRIPTPATH, ver))
        self.input_generator = InputGenerator()

    def close(self):
        tf.reset_default_graph()
        if not self.sess._closed:
            self.sess.close()
        #del self.ops
        #del self.input_generator
        #del self.sess

    def build_graph(self):
        with tf.name_scope('input'): # inputs are aa properties, atom coordinates, distograms
            # AA properties
            aa_in = tf.placeholder(tf.float32, shape=[self.prot_size, 52])
            # 
            # Coordinates
            N  = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            CA = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            C  = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            O  = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            CB = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            C_prev = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            N_next = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, 3])
            atypes_in = tf.placeholder(tf.int8, shape=[None]) # Rosetta atom types (5*nres)
            #
            seqsep = tf.placeholder(tf.float32, shape=[self.prot_size, self.prot_size])
            #
            # distogram (for ver >= 1)
            dist = tf.placeholder(tf.float32, shape=[self.prot_size, self.prot_size, 37])
            #
            mask = tf.placeholder(tf.float32, shape=[self.prot_size, self.prot_size])
            #
            nbatch = tf.shape(N)[0]
            nres = tf.shape(N)[1]
            #
            d_CB = self.calc_distance(CB, CB, transform=True)
            BBtor = self.calc_BBtors(C_prev, N, CA, C, N_next)
            orien = self.calc_6d_transforms(N, CA, C)
            atypes = tf.tile(atypes_in[None,:], (nbatch,1))
            idx, val = self.set_neighbors3D_coarse(N, CA, C, O, CB, atypes)
            #
            tiled_aa = tf.tile(aa_in[None,:,:], (nbatch, 1, 1))
            obt_in = tf.concat((BBtor, tiled_aa), axis=-1)
            del tiled_aa, BBtor
            #
            tiled_seqsep = tf.tile(seqsep[None,:,:,None], (nbatch, 1, 1, 1))
            if self.ver >= 1:
                tiled_disto = tf.tile(dist[None,:,:,:], (nbatch, 1, 1, 1))
                tbt_in = tf.concat((d_CB[:,:,:,None], orien, tiled_seqsep, tiled_disto), axis=-1)
                del tiled_disto
            else:
                tbt_in = tf.concat((d_CB[:,:,:,None], orien, tiled_seqsep), axis=-1)
            del d_CB, orien, tiled_seqsep
            
            # 3D convolution part
            grid3d = tf.scatter_nd(idx, val, [nbatch, nres, 24,24,24,8])
            grid3d = tf.reshape(grid3d, [-1, 24, 24, 24, 8])
            del idx, val
            
            # Training parameters
            dropout_rate = tf.placeholder_with_default(0.15, shape=()) 
            isTraining = tf.placeholder_with_default(False, shape=()) 
        
        layers=[]
        with tf.name_scope('3d_conv'):
            if not self.ignore3dconv:
                # retyper: 1x1x1 convolution
                layers.append(tf.layers.conv3d(grid3d, self.nretype, 1, padding='same', use_bias=False))

                # 1st conv3d & batch_norm & droput & activation
                layers.append(tf.layers.conv3d(layers[-1], 20, 3, padding='valid', use_bias=True))
                layers.append(tf.keras.layers.Dropout(rate=dropout_rate)(layers[-1], training=isTraining))
                layers.append(tf.nn.elu(layers[-1]))

                # 2nd conv3d & batch_norm & activation
                layers.append(tf.layers.conv3d(layers[-1], 30, 4, padding='valid', use_bias=True))
                layers.append(tf.nn.elu(layers[-1]))

                # 3rd conv3d & batch_norm & activation
                layers.append(tf.layers.conv3d(layers[-1], 20, 4, padding='valid', use_bias=True))
                layers.append(tf.nn.elu(layers[-1]))

                # average pooling
                layers.append(tf.layers.average_pooling3d(layers[-1], pool_size=4, strides=4, padding='valid'))
            
        with tf.name_scope('2d_conv'):
            
            # Concat 3dconv output with 1d and project down to 60 dims.
            if not self.ignore3dconv:
                layers.append(tf.reshape(tf.layers.flatten(layers[-1]), [nbatch, nres, 4*4*4*20]))
                layers.append(tf.concat([layers[-1], obt_in], axis=-1))
            else:
                layers.append(tf.concat([obt_in], axis=-1))
            layers.append(tf.nn.elu(tf.layers.conv1d(layers[-1], 60, 1, padding='SAME')))
            
            # Put them together with tbt with self.tbt_size
            tbt = tf.concat([tf.tile(layers[-1][:,:,None,:], [1,1,nres,1]),
                            tf.tile(layers[-1][:,None,:,:], [1,nres,1,1]),
                            tbt_in], axis=-1)
            
            # Do instance normalization after training 
            layers.append(tf.reshape(tbt, [nbatch,nres,nres,self.tbt_size+2*60]))
            layers.append(tf.layers.conv2d(layers[-1], 32, 1, padding='SAME'))
            layers.append(tf.contrib.layers.instance_norm(layers[-1]))
            layers.append(tf.nn.elu(layers[-1]))

            # Resnet prediction with alpha fold style
            resnet_output = build_resnet(layers[-1], 128, self.num_chunks, isTraining)
            layers.append(tf.nn.elu(resnet_output))
            
            # Resnet prediction for errorgram branch
            error_predictor = build_resnet(layers[-1], 128, 1, isTraining)
            error_predictor = tf.nn.elu(error_predictor)
            logits_error = tf.layers.conv2d(error_predictor, filters=15, kernel_size=(1,1))
            estogram_predicted = tf.nn.softmax(logits_error)
            
            # Resnet prediction for errorgram branch
            mask_predictor = build_resnet(layers[-1], 128, 1, isTraining)
            mask_predictor = tf.nn.elu(mask_predictor)
            logits_mask = tf.layers.conv2d(mask_predictor, filters=1, kernel_size=(1,1))[:, :, :, 0]
            mask_predicted = tf.nn.sigmoid(logits_mask) * mask[None,:,:]
            
            # Lddt calculations
            lddt_predicted = self.calculate_LDDT(estogram_predicted, mask_predicted)

        # Exporting out the operaions as dictionary
        return dict(
            aa = aa_in,
            N = N,
            CA = CA,
            C = C,
            O = O,
            CB = CB,
            C_prev = C_prev,
            N_next = N_next,
            mask = mask,
            seqsep = seqsep,
            atypes = atypes_in,
            dist = dist,
            dropout_rate = dropout_rate,
            isTraining = isTraining,
            estogram_predicted = estogram_predicted,
            mask_predicted = mask_predicted,
            lddt_predicted = lddt_predicted,
        )

    # Calculates LDDT based on estogram
    def calculate_LDDT(self, estogram, mask, center=7):
        with tf.name_scope('lddt'):
            # Remove diagonal from calculation
            mask = tf.linalg.set_diag(mask, tf.zeros_like(mask[:,:,0]))
            #mask = tf.multiply(mask, tf.ones(tf.shape(mask))-tf.eye(tf.shape(mask)[1])[None,:,:])
            #masked = tf.transpose(tf.multiply(tf.transpose(estogram, [0,3,1,2]), mask[:,None,:,:]), [0,2,3,1])
            masked = tf.multiply(estogram, mask[:,:,:,None])
            #
            p0 = tf.reduce_sum(masked[:,:,:,center], axis=-1)
            p1 = tf.reduce_sum(masked[:,:,:,center-1]+masked[:,:,:,center+1], axis=-1)
            p2 = tf.reduce_sum(masked[:,:,:,center-2]+masked[:,:,:,center+2], axis=-1)
            p3 = tf.reduce_sum(masked[:,:,:,center-3]+masked[:,:,:,center+3], axis=-1)
            p4 = tf.reduce_sum(mask, axis=-1)

            return 0.25 * (4.0*p0 + 3.0*p1 + 2.0*p2 + p3) / p4
    
    def calc_distance(self, A, B, transform=False):
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
            return f_arcsinh(D)
        else:
            return D

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

    def set_neighbors3D_coarse(self, N, CA, C, O, CB, atypes, d_max=14.0):
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
        dist = self.calc_distance(CA, xyz) # (n_batch, n_res, n_res*5)
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
        return tf.cast(a, dtype=tf.int32), tf.cast(b, dtype=tf.float32)

    def calc_6d_transforms(self, N, CA, C, d_max=20.0):
        nstr = tf.shape(N)[0]
        nres = tf.shape(N)[1]
        mask = tf.ones((nstr, nres, nres), dtype=tf.float32)
        mask = mask - tf.matrix_band_part(mask, 0, 0) # mask diagonal
        Cb = self.get_virtual_CB(N, CA, C)
        # calc 6D transforms for all pairs within d_max
        dist = self.calc_distance(Cb, Cb)
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
        return orien

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
   
    def restore_model(self, ckpt_prefix):
        print ("loading..", ckpt_prefix)
        tot_var_s = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        var_s = list()
        for var in tot_var_s:
            var_s.append(var)
        saver = tf.train.Saver(var_list=var_s)
        if not os.path.exists("%s.index"%ckpt_prefix):
            return False
        saver.restore(self.sess, ckpt_prefix)
        return True

    # Predicting given a batch of information
    def run(self, pose_s, seq_fn=None, seq=None, dist=None, res_ignore=[]):
        aa, seqsep, atypes, batch_s = self.input_generator.process(pose_s, seq_fn=seq_fn, seq=seq, distogram=dist)
        n_res = aa.shape[0] 
        mask = np.ones((n_res, n_res), dtype=np.float32)
        if len(res_ignore) > 0:
            mask[np.ix_(res_ignore, res_ignore)] = 0.0
        lddt_s = list()
        for batch in batch_s:
            N, CA, C, O, CB = batch
            C_prev = np.insert(C, 0, np.nan, axis=1)
            N_next = np.insert(N, n_res, np.nan, axis=1)
           
            if self.ver >= 1:
                feed_dict = {self.ops["aa"]: aa,\
                             self.ops["seqsep"]: seqsep,\
                             self.ops["atypes"]: atypes,\
                             self.ops["N"]: N,\
                             self.ops["CA"]: CA,\
                             self.ops["C"]: C,\
                             self.ops["O"]: O,\
                             self.ops["CB"]: CB,\
                             self.ops["C_prev"]: C_prev,\
                             self.ops["N_next"]: N_next,\
                             self.ops["mask"]: mask,\
                             self.ops["dist"]: dist}
            else:
                feed_dict = {self.ops["aa"]: aa,\
                             self.ops["seqsep"]: seqsep,\
                             self.ops["atypes"]: atypes,\
                             self.ops["N"]: N,\
                             self.ops["CA"]: CA,\
                             self.ops["C"]: C,\
                             self.ops["O"]: O,\
                             self.ops["CB"]: CB,\
                             self.ops["C_prev"]: C_prev,\
                             self.ops["N_next"]: N_next,\
                             self.ops["mask"]: mask}
            
            lddt = self.sess.run(self.ops["lddt_predicted"], feed_dict=feed_dict)
            
            lddt_s.append(lddt)


        lddt = np.concatenate(lddt_s, axis=0)
        #
        return lddt

