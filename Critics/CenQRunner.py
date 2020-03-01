#!/usr/bin/env python

import os
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from CenQ.CenQ_model import CenQPredictor

from pyrosetta import *

class Scorer:
    def __init__(self):
        run_config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        )
        self.CenQ_pred = CenQPredictor(tf.Session(config=run_config))

    def score(self,poses): 
        scores =  self.CenQ_pred.run(poses)
        return scores

if __name__ == "__main__":
    init()
    pdbs =  [l[:-1] for l in open(sys.argv[1])]
    poses = []
    for pdb in pdbs:
        pose = pose_from_file(pdb)
        poses.append(pose)
    scorer = Scorer()
    scores  = scorer.run(pose_s)
    
