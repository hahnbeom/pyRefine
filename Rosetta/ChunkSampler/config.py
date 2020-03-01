import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

CONFIGS = {
    'MINRES_FOR_FRAME': 5, #hard-coded and unusred...
    'SSTYPE_SUPPORTED': ['EE','HEE','HHH'], #['EE','HHH','HH','EEH','EEE'],
    
    'TERM_DBPATH'     : '%s/TERMlib/'%SCRIPTDIR,
    'SEQSCORECUT'     : -1, # everything; use positive value to apply cut
    'MAXLIBRANK'      : {'EE' :10000,
                         'HHH':1000,
                         'HH' :10000,
                         'HEE':1000,
                         'EEE':1000
                          },
    'RMSD_MATCH_CUT'  : 2.0,
    'EXPOSURE_DEF'    : {'E':1,'H':1},
    'CLASH_GRID_BIN'  : 2.0, #angstrom
    'RMSD_ANCHOR_CUT' : 2.0,
    'DUPLICATION_RMSD': 5.0,
    'MAX_CLASH_COUNT' : 1,
    
}
