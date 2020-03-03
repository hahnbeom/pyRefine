#!/usr/bin/env python

import os
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from CenQ.CenQ_model import CenQPredictor

from pyrosetta import *

class Scorer:
    def __init__(self, ver=1):
        run_config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        )
        self.CenQ_pred = CenQPredictor(tf.Session(config=run_config), ver=ver)

    def score(self,poses,seq_fn=None, seq=None, dist=None, res_ignore=[]): 
        scores =  self.CenQ_pred.run(poses, seq_fn=seq_fn, seq=seq, dist=dist, res_ignore=res_ignore)
        return scores

if __name__ == "__main__":
    init()
    pdbs =  [l[:-1] for l in open(sys.argv[1])]
    dist = sys.argv[2]
    poses = []
    import numpy as np
    for pdb in pdbs:
        pose = pose_from_file(pdb)
        poses.append(pose)
    scorer = Scorer()
    scores  = scorer.score(poses, dist=np.load(dist)['dist'].astype(np.float32))
    print (np.mean(scores[0]), np.mean(scores[1]))
    
