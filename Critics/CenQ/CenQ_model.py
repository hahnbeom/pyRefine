#!/usr/bin/env python

import os
import json
import numpy as np
import tensorflow as tf
from .CenQ_resnet import build_resnet
from .InputProcess import InputGenerator
SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))

class CenQPredictor:
    def __init__(self, sess):
        self.sess = sess
        self.prot_size = None
        self.batch_size = None
        self.nretype = 20
        self.num_chunks = 5
        #
        self.obt_size = 56 
        self.tbt_size = 8
        self.ignore3dconv = False
        #
        self.ops = self.build_graph()
        self.restore_model("%s/models_cenQ/model.ckpt"%SCRIPTPATH)

    def build_graph(self):
        with tf.name_scope('input'):
            # 1D convolution part
            obt_in = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, self.obt_size])
            nres = tf.shape(obt_in)[1]
            
            # 2D convolution part
            tbt_in = tf.placeholder(tf.float32, shape=[self.batch_size, self.prot_size, self.prot_size, self.tbt_size])
            self.batch_size = tf.shape(tbt_in)[0]
            
            # 3D convolution part
            idx = tf.placeholder(dtype=tf.int32, shape=(None, 6))
            val = tf.placeholder(dtype=tf.float32, shape=(None))
            grid3d = tf.scatter_nd(idx, val, [self.batch_size, nres, 24,24,24,8])
            grid3d = tf.reshape(grid3d, [-1, 24, 24, 24, 8])
            
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
                layers.append(tf.reshape(tf.layers.flatten(layers[-1]), [self.batch_size, nres, 4*4*4*20]))
                layers.append(tf.concat([layers[-1], obt_in], axis=-1))
            else:
                layers.append(tf.concat([obt_in], axis=-1))
            layers.append(tf.nn.elu(tf.layers.conv1d(layers[-1], 60, 1, padding='SAME')))
            
            # Put them together with tbt with self.tbt_size
            tbt = tf.concat([tf.tile(layers[-1][:,:,None,:], [1,1,nres,1]),
                            tf.tile(layers[-1][:,None,:,:], [1,nres,1,1]),
                            tbt_in], axis=-1)
            
            # Do instance normalization after training 
            layers.append(tf.reshape(tbt, [self.batch_size,nres,nres,self.tbt_size+2*60]))
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
            mask_predicted = tf.nn.sigmoid(logits_mask)
            
            # Lddt calculations
            lddt_predicted = self.calculate_LDDT(estogram_predicted, mask_predicted)

        # Exporting out the operaions as dictionary
        return dict(
            obt = obt_in,
            tbt = tbt_in,
            idx = idx,
            val = val,
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
    def run(self, pose_s, seq_fn=None, seq=None):
        input_generator = InputGenerator(self.sess)
        batch_s = input_generator.process(pose_s, seq_fn=None, seq=None)

        lddt_s = list()
        for batch in batch_s:
            f3d, f1d, f2d = batch
            
            if self.ignore3dconv:
                feed_dict = {self.ops["obt"]: f1d,\
                             self.ops["tbt"]: f2d}
            else:
                feed_dict = {self.ops["obt"]: f1d,\
                             self.ops["tbt"]: f2d,\
                             self.ops["idx"]: f3d[0],\
                             self.ops["val"]: f3d[1]}
            
            operations = [self.ops["estogram_predicted"], 
                          self.ops["mask_predicted"],
                          self.ops["lddt_predicted"]]
           
            estogram, mask, lddt = self.sess.run(operations, feed_dict=feed_dict)
            
            lddt_s.append(lddt)


        lddt = np.concatenate(lddt_s, axis=0)
        #
        return lddt

