#!/bin/bash

pdb=$1
outpdb=$2
script=$3

$ROSETTAPATH/source/bin/relax.linuxgccrelease \
    -s $pdb \
    -relax:script $script \
    -score:weights ref2015_cart \
    -constrain_relax_to_start_coords \
    -out:file:scorefile /dev/null  >&/dev/null

mv ${pdb:0:-4}_0001.pdb $outpdb
