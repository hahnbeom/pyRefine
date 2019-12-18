#!/usr/bin/env python3
import tensorflow as tf

def batch_norm(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)

def inst_norm(inputs):
    return tf.contrib.layers.instance_norm(inputs)

def reweight_seq(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff # msa1hot.shape[1] == n_res
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.log(f_i), axis=1)    
    return tf.concat([f_i, h_i[:,None]], axis=1)
