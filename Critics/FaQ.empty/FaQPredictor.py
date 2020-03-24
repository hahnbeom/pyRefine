import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import pyErrorPred

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#script_dir = os.path.dirname(__file__)
#sys.path.insert(0, script_dir)

def main():
    base = os.path.join(script_dir, "models/")
    modelpath = base + "smTr"
        
    # Featurization happens here #
    inputs = [join(args.infolder, s)+".pdb" for s in samples]
    tmpoutputs = [join(args.outfolder, s)+".features.npz" for s in samples]
    arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
    already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]

    pool = multiprocessing.Pool(num_process)
    out = pool.map(pyErrorPred.process, arguments)

    print(modelpath)

    # Prediction happens here #
    samples = [s for s in samples if isfile(join(args.outfolder, s+".features.npz"))]
    pyErrorPred.predict(samples,
                        np.load(args.distogram)['dist'].astype(np.float32),
                        modelpath,
                        args.outfolder,
                        verbose=args.verbose)

    pyErrorPred.clean(samples,args.outfolder, verbose=args.verbose)
            
if __name__== "__main__":
    main()
