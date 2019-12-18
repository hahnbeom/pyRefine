#!/usr/bin/env python

import os
import sys
import tensorflow as tf
SCRIPTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,SCRIPTPATH+'/SSpred/')
from ResNet_model import ResNet_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_opts(argv):
    import argparse
    opt = argparse.ArgumentParser\
          (description='''AngPred: Backbone torsion angle prediction using ResNet''')
    #required
    opt.add_argument('-a3m_fn', '--a3m_fn', dest='a3m_fn', metavar='a3m_fn', required=True,\
                     help='input a3m file for prediction')
    opt.add_argument('-pdb_fn', '--pdb_fn', dest='pdb_fn', metavar='pdb_fn', required=True,\
                     help='input pdb file for prediction')
    ### optional args
    opt.add_argument('-batch_size', '--batch_size', dest='batch_size', metavar='batch_size', type=int, default=1, \
                     help='The size of batch images [1]')
    opt.add_argument('-n_layer', '--n_layer', dest='n_layer', metavar='n_layer', type=int, default=20, \
                     help='The number of 2D ResNet layers [20]')
    opt.add_argument('-n_1d_layer', '--n_1d_layer', dest='n_1d_layer', metavar='n_1d_layer', type=int, default=12, \
                     help='The number of 1D ResNet layers [12]')
    opt.add_argument('-n_feat', '--n_feat', dest='n_feat', metavar='n_feat', type=int, default=64, \
                     help='The number of hidden features [64]')
    opt.add_argument('-n_bottle', '--n_bottle', dest='n_bottle', metavar='n_bottle', type=int, default=32, \
                     help='The number of hidden bottleures in bottleneck [32]')
    opt.add_argument('-dilation', '--dilation', dest='dilation', metavar='N_dilation', nargs='+', type=int, default=[1, 2, 4, 8], \
                     help='dilation rate for conv [1, 2, 4, 8]')
    opt.add_argument('-model_dir', '--model_dir', dest='model_dir', metavar='model_dir', default='/home/minkbaek/for/hpark/SS_pred/model', \
                     help='directory for checkpoint')
    opt.add_argument('-write_file', dest='write_file', default=True, action="store_true",
                     help='write output as numpy files')
    
    if len(argv) == 1:
        opt.print_help()
        return
    params = opt.parse_args()
    return params

class Predictor:
    def __init__(self):
        return
    
    def run(self,opt=None):
        if opt == None:
            opt = parse_opts(sys.argv[1:])
            
        # sanity check
        if not os.path.exists(opt.pdb_fn):
            sys.exit("No pdb file found! %s"%opt.pdb_fn)
        if not os.path.exists(opt.a3m_fn):
            sys.exit("No a3m file found! %s"%opt.a3m_fn)
        
        run_config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        )
        with tf.Session(config=run_config) as sess:
            print("Built with GPU:", tf.test.is_built_with_cuda())
            print("GPU available:", tf.test.is_gpu_available())
            print("GPU device:", tf.test.gpu_device_name())

            ML_model = ResNet_model(sess,
                                    n_layer=opt.n_layer,
                                    n_1d_layer=opt.n_1d_layer,
                                    n_feat=opt.n_feat,
                                    n_bottle=opt.n_bottle,
                                    dilation=opt.dilation)

            ML_model.predict(opt)

        # return numpy objects
        return ML_model.SS_prob, ML_model.SS3_prob, ML_model.tors_prob, ML_model.dist_s
            

if __name__ == '__main__':
    a = Predictor()
    a.run()

