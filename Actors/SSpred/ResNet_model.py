#!/usr/bin/env python

import os
import time
import tensorflow as tf
import numpy as np
from utils import *
from data_loader import *

N_AA = 20 # regular aa
N_AA_MSA = 21 # regular aa + gap
WMIN = 0.8

N_PRINT_LEVEL = 50

TRAIN_LOG = "Train [%03d/%03d] counter: %5d  time: %10.1f | loss: %7.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
VALID_LOG = "Valid [%03d/%03d] counter: %5d  time: %10.1f | loss: %7.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"

# ResNet model definition
class ResNet_model(object):
    def __init__(self, sess, batch_size=4, n_layer=8, n_1d_layer=2, dilation=[1], 
                 p_dropout=0.2, l2_coeff=0.001, SS_dim=8, phi_dim=36, psi_dim=36, omg_dim=1,
                 n_feat=64, n_bottle=32):
        self.sess = sess # tensorflow session
        #
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.n_1d_layer = n_1d_layer
        self.dilation = dilation
        #
        self.n_feat_1D = 20+22+6
        self.n_str_feat = 7
        self.SS_dim     = SS_dim
        self.phi_dim    = phi_dim
        self.psi_dim    = psi_dim
        self.omg_dim    = omg_dim
        self.tot_dim = SS_dim + phi_dim + psi_dim + omg_dim
        #
        self.n_feat = n_feat
        self.n_bottle = n_bottle
        #
        self.p_dropout = p_dropout
        self.l2_coeff = l2_coeff
        #
        self.build_model()

    def build_model(self):
        # Receive inputs
        with tf.name_scope("input"):
            self.n_res    = tf.placeholder(tf.int32, name="n_res") 
            self.seq      = tf.placeholder(tf.uint8, [None], name="seq") # n_res
            self.msa      = tf.placeholder(tf.uint8, [None, None], name="msa") # n_seq, n_res
            self.bb_angs  = tf.placeholder(tf.float32, [None, 6], name="bb_angs") # n_res, n_angs
            self.str_feat = tf.placeholder(tf.float32, [None, None, self.n_str_feat], name="str_feat") # n_res, n_res, n_feat
            self.SS       = tf.placeholder(tf.float32, [None, None, self.SS_dim]) # SS label
            self.phi      = tf.placeholder(tf.float32, [None, None, self.phi_dim]) # ref. phi distrib
            self.psi      = tf.placeholder(tf.float32, [None, None, self.psi_dim]) # ref. psi distrib
            self.omg      = tf.placeholder(tf.float32, [None, None]) # ref. omg distrib
            self.dist_cb  = tf.placeholder(tf.float32, [None, None, None, 32])
            self.dist_cbt = tf.placeholder(tf.float32, [None, None, None, 32])
            self.dist_cat = tf.placeholder(tf.float32, [None, None, None, 32])
            self.dist_tip  = tf.placeholder(tf.float32, [None, None, None, 32])
            self.mask     = tf.placeholder(tf.float32, [None, None])
            self.is_train = tf.placeholder(tf.bool)
        #
        #================================
        # sequence features
        #================================
        # one-hot encoded seq
        seq1hot = tf.one_hot(self.seq, N_AA, dtype=tf.float32) 
        #
        # get pssm features from MSA
        msa1hot = tf.one_hot(self.msa, N_AA_MSA, dtype=tf.float32)
        w_seq = reweight_seq(msa1hot, WMIN)
        pssm = msa2pssm(msa1hot, w_seq)
        #
        # tiling 1D features
        feat_1D = tf.concat([seq1hot, pssm, self.bb_angs], axis=-1)
        feat_1D = tf.tile(tf.reshape(feat_1D, [-1]), [self.n_res])
        feat_1D = tf.reshape(feat_1D, [self.n_res, self.n_res, self.n_feat_1D])
        #
        # combine with 2D str features
        feat = tf.concat((self.str_feat, feat_1D, tf.transpose(feat_1D, (1,0,2))), axis=-1)
        feat = tf.expand_dims(feat, 0)
        #
        #=================================
        # 2D ResNet extracting str features
        #=================================
        # Initial conv2d with kernel_size 1
        feat = tf.layers.conv2d(feat, self.n_feat, 1, padding='same', use_bias=False)
        
        # Stacking residual blocks
        for i in range(self.n_layer):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block(feat, self.is_train, step=i, dilation=d)
        feat = tf.nn.relu(inst_norm(feat))
        #
        dist_logits = tf.layers.conv2d(feat, 32*4, 1, padding="same", use_bias=False) # for distance prediction
        #
        # convert to 1D using reduce_mean (avg pooling)
        feat = tf.reduce_mean(feat, axis=2)
        
        #=================================
        # 1D ResNet with combined features
        #=================================
        # concatenate with sequence features again & retype
        feat = tf.concat([tf.expand_dims(seq1hot, 0), tf.expand_dims(pssm, 0), feat], axis=-1)
        feat = tf.layers.conv1d(feat, self.n_feat, 1, padding='same', use_bias=False)
        #
        # Stacking 1-dim residual blocks
        for i in range(self.n_1d_layer):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
        feat = tf.nn.relu(inst_norm(feat))
        #
        logits = tf.layers.conv1d(feat, self.tot_dim, 1, padding='same')
        SS_logit = logits[:,:,:self.SS_dim]
        idx = self.SS_dim
        phi_logit = logits[:,:,idx:idx+self.phi_dim]
        idx += self.phi_dim
        psi_logit = logits[:,:,idx:idx+self.psi_dim]
        idx += self.psi_dim
        omg_logit = logits[:,:,idx]
        #
        dist_cb_logit  = dist_logits[:,:,:,  :32]
        dist_cbt_logit = dist_logits[:,:,:,32:64]
        dist_cat_logit = dist_logits[:,:,:,64:96]
        dist_tip_logit = dist_logits[:,:,:,96:]
        
        # calculate loss function (softmax cross-entropy)
        # For SS & omega, it is same as categorical cross entropy
        # For phi psi angles, reference is defined with von Mises distrib.
        # It should be noted that minimizing softmax cross-entropy is same as minimizing KL divergence
        mask_dist = tf.ones((self.n_res, self.n_res), dtype=tf.float32)
        mask_dist = tf.linalg.set_diag(mask_dist, tf.zeros(self.n_res))
        mask_dist = tf.expand_dims(mask_dist, 0)
        mask_dist_mass = tf.reduce_sum(mask_dist)
        #
        mask_mass = tf.reduce_sum(self.mask)
        
        SS_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.SS, SS_logit)
        SS_loss = tf.multiply(self.mask, SS_loss) # masked
        SS_loss = tf.divide(tf.reduce_sum(SS_loss), mask_mass)
        
        phi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.phi, phi_logit)
        phi_loss = tf.multiply(self.mask, phi_loss) # masked
        phi_loss = tf.divide(tf.reduce_sum(phi_loss), mask_mass)

        psi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.psi, psi_logit)
        psi_loss = tf.multiply(self.mask, psi_loss) # masked
        psi_loss = tf.divide(tf.reduce_sum(psi_loss), mask_mass)

        omg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.omg, logits=omg_logit)
        omg_loss = tf.multiply(self.mask, omg_loss) # masked
        omg_loss = tf.divide(tf.reduce_sum(omg_loss), mask_mass)

        dist_cb_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.dist_cb, dist_cb_logit)
        dist_cb_loss = tf.multiply(mask_dist, dist_cb_loss)
        dist_cb_loss = tf.divide(tf.reduce_sum(dist_cb_loss), mask_dist_mass)
        
        dist_cbt_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.dist_cbt, dist_cbt_logit)
        dist_cbt_loss = tf.multiply(mask_dist, dist_cbt_loss)
        dist_cbt_loss = tf.divide(tf.reduce_sum(dist_cbt_loss), mask_dist_mass)
        
        dist_cat_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.dist_cat, dist_cat_logit)
        dist_cat_loss = tf.multiply(mask_dist, dist_cat_loss)
        dist_cat_loss = tf.divide(tf.reduce_sum(dist_cat_loss), mask_dist_mass)
        
        dist_tip_loss = tf.nn.softmax_cross_entropy_with_logits_v2(self.dist_tip, dist_tip_logit)
        dist_tip_loss = tf.multiply(mask_dist, dist_tip_loss)
        dist_tip_loss = tf.divide(tf.reduce_sum(dist_tip_loss), mask_dist_mass)

        #trainable variables
        self.t_vars = tf.trainable_variables()
        #
        l2_loss = tf.add_n([tf.nn.l2_loss(var)
                            for var in self.t_vars if 'kernel' in var.name]) * self.l2_coeff
        #
        # total losses
        self.loss_s = [SS_loss, phi_loss, psi_loss, omg_loss, dist_cb_loss, dist_cbt_loss, dist_cat_loss, dist_tip_loss, l2_loss]
        self.tot_loss = tf.add_n(self.loss_s)
        #
        # calculate probability
        SS_prob = tf.nn.softmax(SS_logit)
        phi_prob = tf.nn.softmax(phi_logit)
        psi_prob = tf.nn.softmax(psi_logit)
        omg_prob = tf.nn.sigmoid(omg_logit)
        dist_cb_prob = tf.nn.softmax(dist_cb_logit)
        dist_cbt_prob = tf.nn.softmax(dist_cbt_logit)
        dist_cat_prob = tf.nn.softmax(dist_cat_logit)
        dist_tip_prob = tf.nn.softmax(dist_tip_logit)
        self.prob_s = [SS_prob, phi_prob, psi_prob, omg_prob, dist_cb_prob, dist_cbt_prob, dist_cat_prob, dist_tip_prob]
        
        # calculate accuracy
        equal = tf.cast(tf.equal(tf.argmax(SS_prob, axis=-1), tf.argmax(self.SS, axis=-1)), tf.float32)
        equal = tf.multiply(self.mask, equal)
        SS_acc = tf.divide(tf.reduce_sum(equal), mask_mass)
        #
        phi_equal = tf.equal(tf.argmax(phi_prob, axis=-1), tf.argmax(self.phi, axis=-1))
        equal = tf.multiply(self.mask, tf.cast(phi_equal, tf.float32))
        phi_acc = tf.divide(tf.reduce_sum(equal), mask_mass)
        #
        psi_equal = tf.equal(tf.argmax(psi_prob, axis=-1), tf.argmax(self.psi, axis=-1))
        equal = tf.multiply(self.mask, tf.cast(psi_equal, tf.float32))
        psi_acc = tf.divide(tf.reduce_sum(equal), mask_mass)
        #
        omg_equal = tf.equal(tf.round(omg_prob), self.omg)
        equal = tf.multiply(self.mask, tf.cast(omg_equal, tf.float32))
        omg_acc = tf.divide(tf.reduce_sum(equal), mask_mass)
        #
        dist_cb_equal = tf.equal(tf.argmax(dist_cb_prob, axis=-1), tf.argmax(self.dist_cb, axis=-1))
        equal = tf.multiply(mask_dist, tf.cast(dist_cb_equal, tf.float32))
        dist_cb_acc = tf.divide(tf.reduce_sum(equal), mask_dist_mass)
        #
        dist_cbt_equal = tf.equal(tf.argmax(dist_cbt_prob, axis=-1), tf.argmax(self.dist_cbt, axis=-1))
        equal = tf.multiply(mask_dist, tf.cast(dist_cbt_equal, tf.float32))
        dist_cbt_acc = tf.divide(tf.reduce_sum(equal), mask_dist_mass)
        #
        dist_cat_equal = tf.equal(tf.argmax(dist_cat_prob, axis=-1), tf.argmax(self.dist_cat, axis=-1))
        equal = tf.multiply(mask_dist, tf.cast(dist_cat_equal, tf.float32))
        dist_cat_acc = tf.divide(tf.reduce_sum(equal), mask_dist_mass)
        #
        dist_tip_equal = tf.equal(tf.argmax(dist_tip_prob, axis=-1), tf.argmax(self.dist_tip, axis=-1))
        equal = tf.multiply(mask_dist, tf.cast(dist_tip_equal, tf.float32))
        dist_tip_acc = tf.divide(tf.reduce_sum(equal), mask_dist_mass)
        #
        equal = tf.stack([phi_equal, psi_equal, omg_equal], axis=-1)
        equal = tf.reduce_all(equal, axis=-1)
        equal = tf.multiply(self.mask, tf.cast(equal, tf.float32))
        tot_acc = tf.divide(tf.reduce_sum(equal), mask_mass)
        self.acc_s = [SS_acc, phi_acc, psi_acc, omg_acc, tot_acc, dist_cb_acc, dist_cbt_acc, dist_cat_acc, dist_tip_acc]
        #
        # saver
        self.saver = tf.train.Saver()
        
    def ResNet_block(self, x, is_train, step=0, dilation=1): # bottleneck block w/ pre-activation
        with tf.variable_scope("ResNet_{}".format(step)) as scope:
            shortcut = x
            # bottleneck layer (kernel: 1, n_feat => n_bottle)
            x = tf.nn.relu(inst_norm(x))
            x = tf.layers.conv2d(x, self.n_bottle, 1, padding='same')
            x = tf.nn.relu(inst_norm(x))
            # convolution
            x = tf.layers.conv2d(x, self.n_bottle, 3, dilation_rate=dilation,
                                 padding='same')
            x = tf.nn.relu(inst_norm(x))
            x = tf.layers.dropout(x, rate=self.p_dropout, training=is_train)
            # project up (kernel: 1, n_bottle => n_feat)
            x = tf.layers.conv2d(x, self.n_feat, 1, padding='same')
            #
            # add
            x += shortcut
        return x
    
    def ResNet_block_1d(self, x, is_train, step=0, dilation=1): # bottleneck block w/ pre-activation
        with tf.variable_scope("ResNet_1d_{}".format(step)) as scope:
            shortcut = x
            # bottleneck layer (kernel: 1, n_feat => n_bottle)
            x = tf.nn.relu(inst_norm(x))
            x = tf.layers.conv1d(x, self.n_bottle, 1, padding='same')
            x = tf.nn.relu(inst_norm(x))
            # convolution
            x = tf.layers.conv1d(x, self.n_bottle, 3, dilation_rate=dilation,
                                 padding='same')
            x = tf.nn.relu(inst_norm(x))
            x = tf.layers.dropout(x, rate=self.p_dropout, training=is_train)
            # project up (kernel: 1, n_bottle => n_feat)
            x = tf.layers.conv1d(x, self.n_feat, 1, padding='same')
            # add
            x += shortcut
        return x
    
    def save(self, folder, prefix):
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.saver.save(self.sess, folder+"/%s.ckpt"%prefix)
    
    def load(self, folder, prefix):
        model_fn = os.path.join(folder, "%s.ckpt.index"%prefix)
        if os.path.exists(model_fn):
            self.saver.restore(self.sess, folder+"/%s.ckpt"%prefix)
            return True
        return False

    def close(self):
        self.sess.close()

    def train(self, config):
        # to update moving mean & variance caused by batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(config.lr)
            optim     = optimizer.minimize(self.tot_loss, var_list=self.t_vars)
        #
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        ops_to_run = [optim, self.tot_loss, self.loss_s, self.acc_s]
        #
        # initialize all variables
        self.sess.run(init_op)
        #
        counter = 0
        self.start_time = time.time()
        #
        train_info = dict([tuple(line.split()) for line in open(config.train_list)])
        n_batch = len(train_info.keys()) 
        min_val_loss = config.best_loss
        #
        # Try to load pre-trained model if exists
        could_load = self.load("model_%d_%d"%(self.n_layer, self.n_1d_layer), 'last_epoch')
        if not could_load:
            could_load = self.load("model_%d_%d"%(self.n_layer, self.n_1d_layer), 'model')
        #
        for epoch in range(config.n_epoch):
            train_pdbs = list(train_info.keys())
            np.random.shuffle(train_pdbs)
            tot_loss_value = 0.0
            tot_loss_s = np.zeros(9, dtype=np.float32)
            tot_acc_s = np.zeros(9, dtype=np.float32)
            n_tot = 0.0
            #
            for pdb in train_pdbs:
                seq, msa, bb_angs, str_feat, SS_labels, phi_labels, psi_labels, omg_labels,\
                        dist_cb, dist_cbt, dist_cat, dist_tip, \
                        masks = load_train_data(pdb, train_info[pdb], random_pick=True)
                #
                _, loss_value, loss_s, acc_s = self.sess.run(ops_to_run,
                                         feed_dict={
                                             self.n_res: len(seq),
                                             self.seq: seq,
                                             self.msa: msa,
                                             self.bb_angs: bb_angs,
                                             self.str_feat: str_feat,
                                             self.SS: SS_labels[np.newaxis,:,:],
                                             self.phi: phi_labels[np.newaxis,:,:],
                                             self.psi: psi_labels[np.newaxis,:,:],
                                             self.omg: omg_labels[np.newaxis,:],
                                             self.dist_cb: dist_cb[np.newaxis,:,:,:],
                                             self.dist_cbt: dist_cbt[np.newaxis,:,:,:],
                                             self.dist_cat: dist_cat[np.newaxis,:,:,:],
                                             self.dist_tip: dist_tip[np.newaxis,:,:,:],
                                             self.mask: masks[np.newaxis,:],
                                             self.is_train: True})
                tot_loss_value += loss_value*float(train_info[pdb])/100.0
                tot_loss_s += np.array(loss_s)*float(train_info[pdb])/100.0
                tot_acc_s += np.array(acc_s)*float(train_info[pdb])/100.0
                n_tot += float(train_info[pdb])/100.0
                #
                counter += 1
                if counter % N_PRINT_LEVEL == 0:
                    loss_value = tot_loss_value/n_tot
                    loss_s = tot_loss_s/n_tot
                    acc_s = tot_acc_s/n_tot
                    tot_loss_value = 0.0
                    tot_loss_s = np.zeros(9, dtype=np.float32)
                    tot_acc_s = np.zeros(9, dtype=np.float32)
                    n_tot = 0.0
                    log_list = [epoch, config.n_epoch, counter, time.time()-self.start_time, loss_value-loss_s[-1]]
                    log_list.extend(loss_s)
                    log_list.extend(acc_s)
                    print (TRAIN_LOG%tuple(log_list))
            #
            val_loss = self.validation(config, epoch, counter)
            if val_loss < min_val_loss:
                self.save("model_%d_%d"%(self.n_layer, self.n_1d_layer), 'model')
                min_val_loss = val_loss
            self.save("model_%d_%d"%(self.n_layer, self.n_1d_layer), 'last_epoch')

    def validation(self, config, epoch, counter):
        ops_to_run = [self.tot_loss, self.loss_s, self.acc_s]
        valid_info = dict([tuple([line.split()[0], line.split()[-1]]) for line in open(config.valid_list) if line[0] != "#"])
        #
        tot_loss_value = 0.0
        tot_loss_s = np.zeros(9, dtype=np.float32)
        tot_acc_s = np.zeros(9, dtype=np.float32)
        n_tot = 0.0
        valid_pdbs = list(valid_info.keys())
        for pdb in valid_pdbs:
            seq, msa, bb_angs, str_feat, SS_labels, phi_labels, psi_labels, omg_labels,\
                    dist_cb, dist_cbt, dist_cat, dist_tip, \
                    masks = load_train_data(pdb, valid_info[pdb], random_pick=True)
            #
            loss_value, loss_s, acc_s = self.sess.run(ops_to_run,
                                             feed_dict={
                                                 self.n_res: len(seq),
                                                 self.seq: seq,
                                                 self.msa: msa,
                                                 self.bb_angs: bb_angs,
                                                 self.str_feat: str_feat,
                                                 self.SS: SS_labels[np.newaxis,:,:],
                                                 self.phi: phi_labels[np.newaxis,:,:],
                                                 self.psi: psi_labels[np.newaxis,:,:],
                                                 self.omg: omg_labels[np.newaxis,:],
                                                 self.dist_cb: dist_cb[np.newaxis,:,:,:],
                                                 self.dist_cbt: dist_cbt[np.newaxis,:,:,:],
                                                 self.dist_cat: dist_cat[np.newaxis,:,:,:],
                                                 self.dist_tip: dist_tip[np.newaxis,:,:,:],
                                                 self.mask: masks[np.newaxis,:],
                                                 self.is_train:False})
            tot_loss_value += loss_value*float(len(seq))/100.0
            tot_loss_s += np.array(loss_s)*float(len(seq))/100.0
            tot_acc_s += np.array(acc_s)*float(len(seq))/100.0
            n_tot += float(len(seq))/100.0
        #
        loss_value = tot_loss_value/n_tot
        loss_s = tot_loss_s/n_tot
        acc_s = tot_acc_s/n_tot
        #
        log_list = [epoch, config.n_epoch, counter, time.time()-self.start_time, loss_value-loss_s[-1]]
        log_list.extend(loss_s)
        log_list.extend(acc_s)
        print (VALID_LOG%tuple(log_list))
        return loss_value-loss_s[-1]
    
    def predict(self, config):
        self.load(config.model_dir, 'model')
        #
        seq, msa, bb_angs, str_feat = make_input_features(config.a3m_fn, config.pdb_fn)
        #
        prob_s = self.sess.run(self.prob_s, feed_dict={
                                                 self.n_res: len(seq),
                                                 self.seq: seq,
                                                 self.msa: msa,
                                                 self.bb_angs: bb_angs,
                                                 self.str_feat: str_feat,
                                                 self.is_train:False})

        self.SS_prob = prob_s[0].reshape(-1, self.SS_dim)
        self.SS3_prob = np.zeros((len(self.SS_prob),3))

        #BEGHIST_: H 2,3,4; E 0,1; L 5,6,7
        self.SS3_prob[:,0] = np.sum([self.SS_prob[:,2],self.SS_prob[:,3],self.SS_prob[:,4]],axis=0)
        self.SS3_prob[:,1] = np.sum([self.SS_prob[:,0],self.SS_prob[:,1]],axis=0)
        self.SS3_prob[:,2] = np.sum([self.SS_prob[:,5],self.SS_prob[:,6],self.SS_prob[:,7]],axis=0)

        phi_prob = prob_s[1].reshape(-1, self.phi_dim)
        psi_prob = prob_s[2].reshape(-1, self.psi_dim)
        omgT_prob = prob_s[3].reshape(-1, self.omg_dim)
        omgC_prob = 1-omgT_prob
        self.tors_prob = np.zeros((len(self.SS_prob),74))
        self.tors_prob[:,:36]   = phi_prob[:,:]
        self.tors_prob[:,36:72] = psi_prob[:,:]
        self.tors_prob[:,72:73] = omgC_prob[:,:] 
        self.tors_prob[:,73:74] = omgT_prob[:,:]     
        self.dist_s  = np.stack([prob_s[i].reshape(len(seq), len(seq), 32) for i in range(4, len(prob_s))], axis=0)
        self.dist_s = self.dist_s.astype(np.float16)
        
        if config.report_files:
            np.save('SS_prob',  self.SS_prob)
            np.save('SS3_prob', self.SS3_prob)
            np.save('tors_prob',self.tors_prob)
            np.save('dist_map', self.dist_s)

