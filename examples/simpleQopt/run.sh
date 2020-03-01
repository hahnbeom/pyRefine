#!/bin/bash

#extra=$1
extra=" -ulr 10-17 32-45"

python ~/NextGenSampler/pyrosetta/Rosetta/test_pyRefineQ.py \
       -s init.pdb \
       -frag_fn_small t000_.3mers \
       -frag_fn_big t000_.9mers \
       -native native.pdb \
       -cen_only \
       -score Q $extra
